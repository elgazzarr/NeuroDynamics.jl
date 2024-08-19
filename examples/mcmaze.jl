# # Example: Latent UDE on the Mcmaze dataset
using Pkg, Revise, Lux, LuxCUDA, Random, DifferentialEquations, SciMLSensitivity, ComponentArrays, Plots, MLUtils, OptimizationOptimisers, LinearAlgebra, Statistics, Printf, PyCall, Distributions, BenchmarkTools, Zygote
using IterTools: ncycle
using NeuroDynamics
np = pyimport("numpy");

device = "gpu"
const dev = device == "gpu" ? gpu_device() : cpu_device()

# ## Loading the data & Creating the dataloaders
file_path = "/home/artiintel/ahmelg/Datasets/mc_maze.npy" # change this to the path of the dataset
data = np.load(file_path, allow_pickle=true)
Y = permutedims(get(data[1], "spikes") , [3, 2, 1]) .|>Float32 |> dev 
n_neurons , n_timepoints, n_trials = size(Y);
ts = range(0.0f0, 5.0f0, length=n_timepoints) |> Array{Float32};
Y_trainval , Y_test = splitobs(Y; at=0.8)
Y_train , Y_val = splitobs(Y_trainval; at=0.8);
train_loader = DataLoader((Y_train, Y_train), batchsize=256, shuffle=true)
val_loader = DataLoader((Y_val, Y_val), batchsize=16, shuffle=true)
test_loader = DataLoader((Y_test, Y_test), batchsize=16, shuffle=true);

#  ## Building the model
hp = Dict("n_states" => 4, "hidden_dim" => 64, "context_dim" => 32, "t_init" => Int(0.5 * n_timepoints))
rng = Random.MersenneTwister(2)
obs_encoder = Recurrent_Encoder(n_neurons, hp["n_states"], hp["context_dim"], 32, hp["t_init"])
drift =  ModernWilsonCowan(hp["n_states"], 0) #Dense(hp["n_states"],hp["n_states"], tanh) #ModernWilsonCowan(hp["n_states"], 0)
drift_aug = Chain(Dense(hp["n_states"] + hp["context_dim"], hp["hidden_dim"], softplus), Dense(hp["hidden_dim"], hp["n_states"], tanh))
diffusion = Scale(hp["n_states"], sigmoid, init_weight=identity_init(gain=0.1f0))

#dynamics =  SDE(drift, drift_aug, diffusion, EulerHeun(), saveat=ts, dt=ts[2]-ts[1], sensealg=TrackerAdjoint()) #InterpolatingAdjoint(; autojacvec = ZygoteVJP()))

dynamics = ODE(drift, Euler(), dt=(ts[2]-ts[1]), saveat=ts, sensealg=GaussAdjoint(; autojacvec = ZygoteVJP()))

obs_decoder = MLP_Decoder(hp["n_states"], n_neurons, 64, 1, "Poisson")   
ctrl_encoder, ctrl_decoder = NoOpLayer(), NoOpLayer()
model = LatentUDE(obs_encoder, ctrl_encoder, dynamics, obs_decoder, ctrl_decoder)
p, st = Lux.setup(rng, model) 
p = p |> ComponentArray{Float32} |> dev


# ## Training the model
function train(model::LatentUDE, p, st, train_loader, val_loader, epochs, print_every)
    epoch = 0
    L = frange_cycle_linear(epochs+1, 0.5f0, 1.0f0, 1, 0.5f0)
    losses = []
    θ_best = nothing
    best_metric = -Inf
    stime = time()
    @info "Training ...."

   function loss(p, y, _)
        ŷ, _, x̂₀, kl_path = model(y, nothing, ts, p, st)
        batch_size = size(y)[end]
        recon_loss = -poisson_loglikelihood(ŷ, y)/ batch_size
        kl_init = kl_normal(x̂₀[1], x̂₀[2])/batch_size
        #kl_path = mean(kl_path[end,:])
        kl_path = 0.0f0
        kl_loss =  kl_path  +  kl_init
        l =  recon_loss + L[epoch+1]*kl_loss
        return l, recon_loss, kl_loss
    end


    callback = function(opt_state, l, recon_loss , kl_loss)
        θ = opt_state.u
        push!(losses, l)
        if length(losses) % length(train_loader) == 0
            epoch += 1
        end

        if length(losses) % (length(train_loader)*print_every) == 0
            t_epoch = time() - stime
            @printf("Time/epoch %.2fs \t Current epoch: %d, Loss: %.2f, PoissonLL: %d, KL: %.2f\n", t_epoch/print_every, epoch, losses[end], recon_loss, kl_loss)
            y, _ = first(val_loader) 
            ŷ, _, _ = predict(model, y, nothing, ts, θ, st, 20, dev)
            ŷₘ = dropdims(mean(ŷ, dims=4), dims=4)
            val_bps = bits_per_spike(ŷₘ, y)
            @printf("Validation bits/spike: %.2f\n", val_bps)
            if val_bps > best_metric
                best_metric = val_bps
                 θ_best = copy(θ)
                @printf("Saving best model\n")
            end
            stime = time()        
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ , y, y_) -> loss(p, y, y_), adtype)
    optproblem = OptimizationProblem(optf, p)
    result = Optimization.solve(optproblem, ADAMW(1e-3), ncycle(train_loader, epochs); callback)
    return model, θ_best
    
end

model, θ_best = train(model, p, st, train_loader, val_loader, 50, 10);
println("Done training")







