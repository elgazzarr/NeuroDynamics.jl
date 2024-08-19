
using DifferentialEquations, Lux, ComponentArrays, Random, SciMLSensitivity, Zygote, BenchmarkTools, LuxCUDA, CUDA,
OptimizationOptimisers, Optimisers, Printf

dev = cpu_device()

data = rand32(32,100,512) |> dev
x₀ = rand32(32,512) |> dev
ts = range(0.0f0, 1.0f0, length=100)
drift = Dense(32, 32, tanh)
diffusion = Scale(32, sigmoid)

basic_tgrad(u, p, t) = zero(u)

struct NeuralSDE{D, F} <: Lux.AbstractExplicitContainerLayer{(:drift, :diffusion)}
    drift::D
    diffusion::F
    solver
    tspan
    sensealg
end

function (model::NeuralSDE)(x₀, ts, p, st)
    μ(u, p, t) = model.drift(u, p.drift, st.drift)[1]
    σ(u, p, t) = model.diffusion(u, p.diffusion, st.diffusion)[1]
    func = SDEFunction{false}(μ, σ; tgrad=basic_tgrad)
    prob = SDEProblem{false}(func, x₀, model.tspan, p)
    sol = solve(prob, model.solver; saveat=ts, dt=0.01f0, sensealg = model.sensealg)
    return permutedims(cat(sol.u..., dims=3), (1,3,2))
end


struct NeuralODE{D, F} <: Lux.AbstractExplicitContainerLayer{(:drift, :diffusion)}
    drift::D
    diffusion::F
    solver
    tspan
    sensealg
end

function (model::NeuralODE)(x₀, ts, p, st)
    μ(u, p, t) = model.drift(u, p.drift, st.drift)[1]
    σ(u, p, t) = model.diffusion(u, p.diffusion, st.diffusion)[1]
    func = ODEFunction{false}(μ)
    prob = ODEProblem{}(func, x₀, model.tspan, p)
    sol = solve(prob, model.solver; saveat=ts, dt=0.01f0, sensealg = model.sensealg)
    return permutedims(cat(sol.u..., dims=3), (1,3,2))
end



function loss!(model, p, st, (x₀, data))
    pred = model(x₀, ts, p, st)
    l = sum(abs2, data .- pred)
    return l, st, pred
end

rng = Random.default_rng()
model = NeuralSDE(drift, diffusion, EM(), (0.0f0, 1.0f0), BacksolveAdjoint(;autojacvec=ZygoteVJP())) #InterpolatingAdjoint(;autojacvec=ZygoteVJP()))



function train_model!(model, nepochs::Int)
    p, st = Lux.setup(rng, model)
    p = p |> ComponentArray{Float32} |> dev
    st = st |> dev 
    train_state = Lux.Experimental.TrainState(model, p, st,  Adam(0.01f0))
    for i in 1:nepochs
        @time grads, loss, _, train_state = Lux.Experimental.single_train_step!(
            AutoZygote(), loss!, (x₀, data), train_state)
        if i % 10 == 1 || i == nepochs
            @printf "Loss Value after %6d iterations: %.8f\n" i loss
        end
    end
    return train_state.model, train_state.parameters, train_state.states
end

model, ps, st = train_model!(model, 50)

