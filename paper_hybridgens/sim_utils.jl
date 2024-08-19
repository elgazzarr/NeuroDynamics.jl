using Lux, LuxCUDA, Random, DifferentialEquations, SciMLSensitivity, ComponentArrays,
 Plots, MLUtils, OptimizationOptimisers, LinearAlgebra, Statistics, PyCall, Distributions, BenchmarkTools, Zygote
using IterTools: ncycle
using NeuroDynamics
using  FileIO, Printf

function ModernWilsonCowan2(N, M)
    @compact(τ = truncated_normal(N, lo=0, hi=0.5),
             J = orthogonal(N, N),
             B = glorot_uniform(N, M),
             b = ones32(N),
             name="WilsonCowan ($N states, $M inputs)") do xu
        x, u = xu      
        dx = (-x + tanh.(J * x + B * u .+ b))./τ
        @return dx
    end
end 


function generate_data(n_states, n_inputs, n_observations, n_samples, noise_level)
    rng = Random.MersenneTwister(1)
    N = n_states; M = n_inputs;
    tspan = (0.0f0, 5.0f0)
    ts = range(tspan[1], tspan[2], length=50)
    x0 = rand32(N)
    freq = rand32(M); φ = 2π .* rand32(M);
    u(t) = sin.(2π .* freq .* t .+ φ) * 5.0
    vf = ModernWilsonCowan2(N, M)
    p, st = Lux.setup(rng, vf)
    function drift(x, p, t)
        xu = (x, u(t))
        return vf(xu, p, st)[1]
    end
    diffusion(x, p, t) = noise_level*rand32(n_states).*x
    prob = SDEProblem(drift, diffusion, x0, tspan, p)
    prob_f(prob, i, repeat) = remake(prob, u0=rand32(N));
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_f)
    sol = solve(ensemble_prob, SOSRI(),  EnsembleThreads(); saveat=ts, trajectories=n_samples);
    x = Array{Float32}(sol)
    projection = Dense(n_states, n_observations, init_weight=sparse_init(;sparsity=0.6, std=3.0), sigmoid)
    p_W, st_W = Lux.setup(rng, projection)
    r, _ = projection(x, p_W, st_W)
    y = rand.(Poisson.(r))
    display(plot(sol))
    return cat(u.(ts)...,dims=2), sol, r, y, ts
end;

function createloaders(inputs, spikes, rates; batch_size=64)
    U = repeat(inputs, 1, 1, size(spikes, 3)) |> Array{Float32}
    spikes = Array{Float32}(spikes)
    (u_train, y_train, r_train), (u_val, y_val, r_val) = splitobs((U, spikes, rates); at=0.9)    
    train_loader = DataLoader((u_train, y_train, r_train), batchsize=batch_size, shuffle=true)
    val_loader = DataLoader((u_val, y_val, r_val), batchsize=32, shuffle=true)
    return train_loader, val_loader
end

function setup_model(sde::Bool, ts, dims, config, dev)
    rng = Random.MersenneTwister(1)
    obs_encoder = Recurrent_Encoder(dims.n_neurons, config.N, config.C, config.o_encoder.hidden, config.t₀)
    init_map = Dense(config.N, config.N)

    drift = @compact(m=Chain(Dense(config.N + dims.n_stimuli => 64, swish), Dense(64, config.N, tanh))) do xu 
        @return m(vcat(xu...))
    end

    if sde
        drift_aug = Chain(Dense(config.N + config.C + dims.n_stimuli, 64, swish), Dense(64, config.N, tanh))
        diffusion = Scale(config.N, sigmoid, init_weight=identity_init(gain=0.1f0))
        dynamics = SDE(drift, drift_aug, diffusion, EM(), saveat=ts, dt=(ts[2]-ts[1]))
    else 
        dynamics = ODE(drift, Euler(), saveat=ts, dt=(ts[2]-ts[1])*2)
    end

    obs_decoder = Linear_Decoder(config.N, dims.n_neurons, "Poisson")
    ctrl_encoder, ctrl_decoder = NoOpLayer(), NoOpLayer()
    model = LatentUDE(obs_encoder, ctrl_encoder, init_map, dynamics, obs_decoder, ctrl_decoder)
    p, st = Lux.setup(rng, model) 
    p = p |> ComponentArray{Float32} |> dev
    st = st |> dev 
    return model, p, st 
end


