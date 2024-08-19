using Pkg, Revise
using Lux, Random, DifferentialEquations, NeuroDynamics
using FileIO, JLD2
using ComponentArrays, LuxCUDA
using CairoMakie
include("sim_utils.jl")
include("models.jl")


noise_level = 1.5
dev = cpu_device()
tspan = (0.0f0, 5.0f0); ts = range(tspan[1], tspan[2], length=50);
model_config = (N = 5, C = 32, t₀=0.9, o_encoder=(hidden=32,), o_decoder = (hidden=64, depth=1))
dims = (n_neurons = 50, n_behaviour = 0, n_stimuli = 2, n_timepoints = 50, n_trials = 100)
model, θ, st = setup_model(false, ts, dims, model_config, dev);

exp_path = "/home/artiintel/ahmelg/code/NeuroDynamics.jl/paper_hybridgens/results/simulations/latentode-N5-σ$noise_level/bestmodel.jld2"
s = load_object(exp_path);
_, p, st, dataloader, epoch = s;
pll = eval(dataloader, p, st)

u, y, r = first(dataloader);
ŷ, û, x = predict(model, y, u, ts, p, st, 20, cpu_device());



function eval(dataloader, θ, st)
    pll = 0.f0
    for (u, y, _) in dataloader
        u = u; y = y;
        Eŷ, _, x = predict(model, y, u, ts, θ, st, 20, cpu_device())
        pll += mean(mapslices(ŷ -> poisson_loglikelihood(ŷ, y), Eŷ, dims=[1,2,3])) / size(y, 3)
    end
    pll /= length(dataloader)
    return pll
end





function plotrates(ŷ, gt_rates, ts, sample_n, ch)
    gt_rates = gt_rates[ch, :, sample_n]
    fig = Figure(size = (800, 600), backgroundcolor = :transparent)
    ax1 = CairoMakie.Axis(fig[1,1], xlabel = "time(ms)", ylabel = "rates", backgroundcolor = :transparent, limits = (nothing, (0, 1.3)))
    t = ts 
    ŷₘ = selectdim(dropmean(ŷ, dims=4), 3, sample_n)
    ŷₛ = selectdim(dropmean(std(ŷ, dims=4), dims=4), 3, sample_n)
    lines!(ts, gt_rates, color = :red, linewidth = 3, label = "ground truth rates")
    lines!(ts, ŷₘ[ch,:], linewidth = 2, color = (:dodgerblue2, 0.5))
    band!(t, ŷₘ[ch,:] .-   sqrt.(ŷₛ[ch,:]) , ŷₘ[ch,:] .+ sqrt.(ŷₛ[ch,:]), color= (:dodgerblue2, 0.5),  label = "generated rates")
    hidedecorations!(ax1)
    #hideydecorations!(ax1, label = true)
    #hidexdecorations!(ax1, ticklabels = false, label=false)
    axislegend(backgroundcolor = :transparent)
    hidespines!(ax1)
    fig
end



sample_n = 2; ch = 22
fig = plotrates(ŷ, r, ts, sample_n, ch)
save("figures/sim/r_S$sample_n-C$ch.svg", fig)


function plotstates(x, sample_n)
    fig = Figure(size = (800, 600), backgroundcolor = :transparent)
    ax1 = CairoMakie.Axis(fig[1,1], xlabel = "time(ms)", ylabel = "states", backgroundcolor = :transparent)
    t = ts 
    xₘ = selectdim(dropmean(x, dims=4), 3, sample_n)
    xₛ = selectdim(dropmean(std(x, dims=4), dims=4), 3, sample_n)

    for ch in 1:size(x, 1)
        lines!(ts, xₘ[ch,:], linewidth = 2)
        
        band!(t, xₘ[ch,:] .- xₛ[ch,:], xₘ[ch,:] .+ xₛ[ch,:], alpha = 0.3)
    end
    hideydecorations!(ax1, label = true)
    hidexdecorations!(ax1, ticklabels = false, label=false)
    fig
end

fig = plotstates(x, sample_n)

