using Pkg, Revise
using Lux, LuxCUDA, Random, DifferentialEquations, SciMLSensitivity, ComponentArrays,
 Plots, MLUtils, OptimizationOptimisers, LinearAlgebra, Statistics, Printf, PyCall, Distributions, BenchmarkTools, Zygote
using IterTools: ncycle
using NeuroDynamics
using WandbMacros
include("../trainer.jl")

function ModernWilsonCowan2(N, M)
    @compact(τ = rand32(N),
             J = glorot_uniform(N, N),
             B = glorot_uniform(N, M),
             b = ones32(N),
             name="WilsonCowan ($N states, $M inputs)") do xu
        x, u = xu      
        dx = (-x + sigmoid.(J * x + B * u .+ b))./τ
        @return dx
    end
end 


function generate_data(n_states, n_inputs, n_observations, n_samples)
    N = n_states; M = n_inputs;
    tspan = (0.0f0, 10.0f0)
    ts = range(tspan[1], tspan[2], length=100)
    x0 = rand32(N)
    freq = rand32(M); φ = 2π .* rand32(M);
    u(t) = sin.(2π .* freq .* t .+ φ)
    vf = ModernWilsonCowan2(N, M)
    p, st = Lux.setup(rng, vf)
    function drift(x, p, t)
        xu = (x, u(t))
        return vf(xu, p, st)[1]
    end
    diffusion(x, p, t) = 5e-2.*rand32(size(x,1)).*x
    prob = SDEProblem(drift, diffusion, x0, tspan, p)
    prob_f(prob, i, repeat) = remake(prob, u0=rand32(N));
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_f)
    sol = solve(ensemble_prob, SOSRI(),  EnsembleThreads(); saveat=ts, trajectories=n_samples);
    x = Array{Float32}(sol)
    projection = Dense(n_states, n_observations, init_weight=sparse_init(;sparsity=0.2, std=1.0), sigmoid)
    p_W, st_W = Lux.setup(rng, projection)
    r, _ = projection(x, p_W, st_W)
    y = rand.(Poisson.(r))
    display(plot(sol))
    return cat(u.(ts)...,dims=2), x, r, y, ts
end


function createloaders(inputs, spikes, rates; batch_size=64)
    U = repeat(inputs, 1, 1, size(spikes, 3)) |> Array{Float32}
    (u_train, y_train, r_train), (u_val, y_val, r_val) = splitobs((U, spikes, rates); at=0.8)    
    train_loader = DataLoader((u_train, y_train, r_train), batchsize=batch_size, shuffle=true)
    val_loader = DataLoader((u_val, y_val, r_val), batchsize=32, shuffle=true)
    return train_loader, val_loader
end

function setup_model(sde::Bool, ts, dims, config, dev)
    rng = Random.MersenneTwister(2)
    obs_encoder = Recurrent_Encoder(dims.n_neurons, config.N, config.C, config.o_encoder.hidden, config.t₀)
    init_map = Dense(config.N, config.N)

    drift = Chain(Dense(config.N + dims.n_stimuli => 64, softplus),
                Dense(64 => config.N, tanh))#Chain(Dense(config.N + dims.n_stimuli => config.N, tanh))

    if sde
        drift_aug = Chain(Dense(config.N + config.C + dims.n_stimuli, 64, tanh), Dense(64, config.N, tanh))
        diffusion = Scale(config.N, sigmoid, init_weight=identity_init(gain=0.1f0))
        dynamics = SDE(drift, drift_aug, diffusion, EM(), saveat=ts, dt=(ts[2]-ts[1]))
    else 
        dynamics = ODE(drift, Euler(), saveat=ts, dt=(ts[2]-ts[1])*2)
    end

    obs_decoder = Linear_Decoder(config.N, dims.n_neurons, "Poisson") #MLP_Decoder(config.N, dims.n_neurons, config.o_decoder.hidden, config.o_decoder.depth, "Poisson")   
    ctrl_encoder, ctrl_decoder = NoOpLayer(), NoOpLayer()
    model = LatentUDE(obs_encoder, ctrl_encoder, init_map, dynamics, obs_decoder, ctrl_decoder)
    p, st = Lux.setup(rng, model) 
    p = p |> ComponentArray{Float32} |> dev
    st = st |> dev 
    return model, p, st 
end

use_gpu = false
dev = use_gpu ? gpu_device() : cpu_device() 
rng = Random.default_rng()

#Ground truth data
data_config = (n_states=4, n_inputs=2, n_observations=100, n_samples=512)
inputs, states, rates, spikes, ts = generate_data(data_config...);
train_loader, val_loader = createloaders(inputs, spikes, rates; batch_size=256)
dims = (n_neurons = data_config.n_observations, n_behaviour = 0, n_stimuli = data_config.n_inputs, n_timepoints = size(spikes,2), n_trials = data_config.n_samples)

#Model
model_config = (N = 4, C = 32, t₀=50, o_encoder=(hidden=32,), o_decoder = (hidden=64, depth=0))
model, θ, st = setup_model(true, ts, dims, model_config, dev);

#Train
@wandbinit project="HybridGen_Neuro" name="WilsonCowan_test"
train_config = (lr=1e-3, epochs=100, log_freq=10)
model, θ_best = train(model, θ, st, train_loader, val_loader, train_config, ts, dev);


function evaluate(data_loader, θ; sample, ch)
    u, y, r = first(data_loader)
    ŷ, _, x = predict(model, y, u, ts, θ, st, 20, dev)
    ŷₘ = dropdims(mean(ŷ, dims=4), dims=4)
    ŷₛ = dropdims(std(ŷ, dims=4), dims=4)
    dist = Poisson.(ŷₘ)
    pred_spike = rand.(dist)
    xₘ = dropdims(mean(x, dims=4), dims=4)
    @show val_bps = bits_per_spike(ŷₘ, y)
    p1 = plot(transpose(y[ch:ch,:,sample]), label="True Spike", lw=2, color="red", grid=false)
    p2 = plot(transpose(pred_spike[ch:ch,:,sample]), label="Predicted Spike", lw=2, color="green", grid=false)
    p3 = plot(transpose(r[ch:ch,:,sample]), label="true rates", lw=3, color="red")
    p3 = plot!(p3, transpose(ŷₘ[ch:ch,:,sample]), ribbon=transpose(ŷₛ[ch:ch,:,sample]), label="Infered rates", lw=2, color="green", yticks=false)
    p4 = plot_states(x, sample; label=false, yticks=false)
    plot(p1, p2, p3, p4, layout=(4,1), size=(1200, 800), legend=:topright)
end

evaluate(train_loader, θ_best; sample=4, ch=2)