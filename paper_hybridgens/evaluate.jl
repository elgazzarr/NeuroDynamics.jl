using Pkg, Revise
using Lux, Random, DifferentialEquations, NeuroDynamics
using FileIO, JLD2
using ComponentArrays , Statistics, LinearAlgebra
using CairoMakie
include("HybridGens.jl")
using .HybridGens
using Plots


dataset = "mc_maze"
model = "SlOscillatorsHybrid-sde"

exp_path = "/home/artiintel/ahmelg/code/NeuroDynamics.jl/paper_hybridgens/results/model_comparison/$dataset/$model/bestmodel.jld2"
s = load_object(exp_path);
model, p, st, dataloader, epoch = s;
u, y, b = first(dataloader);
ts = range(0.0, 4.0, length = size(y,2))
ŷ, û, x = predict(model, y, u, ts, p, st, 50, cpu_device());
anim = NeuroDynamics.animate_oscillators(x[:,:,5,2])
gif(anim)
#plot_samples(ŷ[10:22,:,:,:],10)

figure = Figure(size=(900, 600))
ax1 = CairoMakie.Axis(figure[1, 1], xlabel = "Time", ylabel = "Hand velocity")
ax2 = CairoMakie.Axis(figure[2, 1], xlabel = "Time", ylabel = "Latent state")
ax3 = CairoMakie.Axis3
lines!(ax1, ts, b[1,:,2], color = :red, label = "x-velocity")
lines!(ax1, ts, b[2,:,2], color = :red, label = "y-velocity")
map(i -> plot_dist(x, i, 1) , 1:div(size(x, 1),2))

figure


function plot_dist(x, ch, sample_n)
    xₘ = selectdim(dropmean(x, dims=4), 3, sample_n)
    xₛ = selectdim(dropmean(std(ŷ, dims=4), dims=4), 3, sample_n)
    lines!(ts, xₘ[ch,:], linewidth = 2)
    band!(ts, xₘ[ch,:] .-  sqrt.(xₛ[ch,:]) , xₘ[ch,:] .+ sqrt.(xₛ[ch,:]), alpha =  0.3)
end


function create_phaseplot(model, p, st, t)
    function sol(x)
        xu = (x, u)
        dx = model(xu, p, st)[1]
        println(size(dx))
        return Point2f(dx...)
    end
    fig = Figure(size = (1200, 900), backgroundcolor = :transparent)
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "x¹", ylabel = "x²", backgroundcolor = :transparent)
    streamplot!(ax, sol, -5 .. 5, -10 .. 10, colormap = Reverse(:viridis),
        gridsize = (64, 64), arrow_size = 10, linewidth = 1.5)
    fig
end

create_phaseplot(model.dynamics.vector_field, p.dynamics, st.dynamics, 0.0)