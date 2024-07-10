# Infering neural dynamics of motor cortex during dealyed reach task using a latent SDE

In this example, we will show how to use the latentsde model to infer underlying neural dynamics from single trial spiking recordings of neurons in the dorsal premotor (PMd) and primary motor (M1) cortices.
The data is available for download [here](https://dandiarchive.org/#/dandiset/000128).

Dynamics in the motor cortext are known to be highly autonomus during simple stereotyped tasks, so it can be predictable given an "informative" initial condition even in the absence of stimulus information. 


```julia
using Pkg, Revise, Lux, LuxCUDA, Random, DifferentialEquations, SciMLSensitivity, ComponentArrays, Plots, MLUtils, OptimizationOptimisers, LinearAlgebra, Statistics, Printf, PyCall, Distributions, BenchmarkTools, Zygote
using IterTools: ncycle
using NeuroDynamics
np = pyimport("numpy");
```


```julia
device = "cpu"
const dev = device == "gpu" ? gpu_device() : cpu_device()

```

## 1. Loading the data and creating the dataloaders

You can prepare the data yourself or use our preprocessed data staright away which is available [here](https://drive.google.com/file/d/1J9)


```julia
file_path = "/Users/ahmed.elgazzar/Datasets/NLB/mc_maze.npy" # change this to the path of the dataset
data = np.load(file_path, allow_pickle=true)
Y = permutedims(get(data[1], "spikes") , [3, 2, 1]) |> Array{Float32}
n_neurons , n_timepoints, n_trials = size(Y) 
ts = range(0, 5.0, length=n_timepoints) |> Array{Float32}
Y_trainval , Y_test = splitobs(Y; at=0.8)
Y_train , Y_val = splitobs(Y_trainval; at=0.8);
train_loader = DataLoader((Y_train, Y_train), batchsize=32, shuffle=true)
val_loader = DataLoader((Y_val, Y_val), batchsize=16, shuffle=true)
test_loader = DataLoader((Y_test, Y_test), batchsize=16, shuffle=true);
```

## 2. Defining the model 
- We will use a "Recurrent_Encoder" to infer the initial hidden state from a portion of the observations. 
- We will use a BlackBox (Neural) SDE with multiplicative noise to model the latent dynamics.
- We will use a decoder with a Poisson likelihood to model the spike counts.


```julia
hp = Dict("n_states" => 10, "hidden_dim" => 64, "context_dim" => 32, "t_init" => Int(0.9 * n_timepoints))
rng = Random.MersenneTwister(2)
obs_encoder = Recurrent_Encoder(n_neurons, hp["n_states"], hp["context_dim"], 32, hp["t_init"])
drift =  ModernWilsonCowan(hp["n_states"], 0)
drift_aug = Chain(Dense(hp["n_states"] + hp["context_dim"], hp["hidden_dim"], softplus), Dense(hp["hidden_dim"], hp["n_states"], tanh))
diffusion = Scale(hp["n_states"], sigmoid, init_weight=identity_init(gain=0.1))
dynamics =  SDE(drift, drift_aug, diffusion, EulerHeun(), saveat=ts, dt=ts[2]-ts[1]) #ODE(drift, Tsit5)
obs_decoder = MLP_Decoder(hp["n_states"], n_neurons, 64, 1, "Poisson")   
ctrl_encoder, ctrl_decoder = NoOpLayer(), NoOpLayer()
model = LatentUDE(obs_encoder, ctrl_encoder, dynamics, obs_decoder, ctrl_decoder, dev)
p, st = Lux.setup(rng, model) .|> dev
p = p |> ComponentArray{Float32} 
```

## 3. Training the model 

We will train the model using the AdamW optimizer with a learning rate of 1e-3 for 200 epochs. 



```julia
function train(model::LatentUDE, p, st, train_loader, val_loader, epochs, print_every)
    epoch = 0
    L = frange_cycle_linear(epochs+1, 0.5f0, 1.0f0, 1, 0.5)
    losses = []
    θ_best = nothing
    best_metric = -Inf
    @info "Training ...."

    function loss(p, y, _)
        y, ts_ = y |> dev, ts |> dev
        ŷ, _, x̂₀, kl_path = model(y, nothing, ts_, p, st)
        batch_size = size(y)[end]
        recon_loss = - poisson_loglikelihood(ŷ, y)/batch_size
        kl_init = kl_normal(x̂₀[1], x̂₀[2])
        kl_path = mean(kl_path[end,:])
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
            @printf("Current epoch: %d, Loss: %.2f, PoissonLL: %d, KL: %.2f\n", epoch, losses[end], recon_loss, kl_loss)
            y, _ = first(val_loader) 
            ŷ, _, _ = predict(model, y, nothing, ts, θ, st, 20)
            ŷₘ = dropdims(mean(ŷ, dims=4), dims=4)
            val_bps = bits_per_spike(ŷₘ, y)
            @printf("Validation bits/spike: %.2f\n", val_bps)
            if val_bps > best_metric
                best_metric = val_bps
                 θ_best = copy(θ)
                @printf("Saving best model")
            end        
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ , y, y_) -> loss(p, y, y_), adtype)
    optproblem = OptimizationProblem(optf, p)
    result = Optimization.solve(optproblem, ADAMW(1e-3), ncycle(train_loader, epochs); callback)
    return model, θ_best
    
end

```


```julia
model, θ_best = train(model, θ_best, st, train_loader, val_loader, 100, 10);
```


```julia
y, _ = first(test_loader)
sample = 24
ch = 4
ŷ, _, x = predict(model, y, nothing, ts, θ_best, st, 20)
ŷₘ = dropdims(mean(ŷ, dims=4), dims=4)
ŷₛ = dropdims(std(ŷ, dims=4), dims=4)
dist = Poisson.(ŷₘ)
pred_spike = rand.(dist)
xₘ = dropdims(mean(x, dims=4), dims=4)
val_bps = bits_per_spike(ŷₘ, y)

p1 = plot(transpose(y[ch:ch,:,sample]), label="True Spike", lw=2)
p2 = plot(transpose(pred_spike[ch:ch,:,sample]), label="Predicted Spike", lw=2, color="red")
p3 = plot(transpose(ŷₘ[ch:ch,:,sample]), ribbon=transpose(ŷₛ[ch:ch,:,sample]), label="Infered rates", lw=2, color="green", yticks=false)

plot(p1, p2,p3, layout=(3,1), size=(800, 400), legend=:topright)

```
