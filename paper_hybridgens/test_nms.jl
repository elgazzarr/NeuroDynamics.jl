using Pkg, Revise
using Lux, Random, DifferentialEquations, NeuroDynamics
using ComponentArrays
using Plots
using Zygote
include("models.jl")
include("hybridmodels.jl")

N = 2; M=3;
rng = Random.default_rng()
ts = range(0.0f0, 10.0f0, length=500)

N_actual = N*2*3
y = rand32(N_actual,length(ts),1)
U = cos.(ts)
u = permutedims(repeat(U, 1, M, 1), (2, 1, 3)) |> Array{Float32}
vf = JensenRit(2,2,2,M)
drift_aug = Dense(N, N, tanh)
diffusion = Scale(N_actual, sigmoid,  init_weight=identity_init(gain=0.1f0))
dynamics = SDE(vf, drift_aug, diffusion, EM();  saveat=ts, dt=0.01f0)
LVM = LatentUDE(dynamics = dynamics)
p, st = Lux.setup(rng, LVM)
p = p |> ComponentArray{Float32}
ŷ, û, x = predict(LVM, y, u, ts, p, st, 20, cpu_device())
plot(transpose(x[1:N,:,1,1]))






N_ex = 4; N_in = 4; M=2;
N = N_ex + N_in
rng = Random.default_rng()
x0=rand32(N,1); u0=rand32(M,1);
y = rand32(N,50,1)
ts = range(0.0f0, 10.0f0, length=100)
U = cos.(ts)
u = permutedims(repeat(U, 1, M, 1), (2, 1, 3)) |> Array{Float32}
vf = WilsonCowan_(N_ex, N_in, M)
dynamics = ODE(vf, Euler();  saveat=ts, dt=0.01f0)
LVM = LatentUDE(dynamics = dynamics)
p, st = Lux.setup(rng, LVM)
p = p |> ComponentArray{Float32}
ŷ, û, x = predict(LVM, y, u, ts, p, st, 20, cpu_device())
plot(transpose(x[1:N,:,1,1]))
