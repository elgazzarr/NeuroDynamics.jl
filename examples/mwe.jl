
using DifferentialEquations, Lux, ComponentArrays, Random, SciMLSensitivity, Zygote, BenchmarkTools, LuxCUDA, CUDA,
OptimizationOptimisers



dev = gpu_device()
sensealg = TrackerAdjoint()

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

function loss!(p, data)
    pred = model(x₀, ts, p, st)
    l = sum(abs2, data .- pred)
    return l, st, pred
end

rng = Random.default_rng()
model = NeuralSDE(drift, diffusion, EM(), (0.0f0, 1.0f0) ,sensealg)
p, st = Lux.setup(rng, model)
p = p |> ComponentArray{Float32} |> dev


adtype = AutoZygote()
optf = OptimizationFunction((p, _ ) -> loss!(p, data), adtype)
optproblem = OptimizationProblem(optf, p)
result = Optimization.solve(optproblem, ADAMW(5e-4), maxiters=10)


