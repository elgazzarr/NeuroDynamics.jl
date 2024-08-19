using CairoMakie


#loglikelhodd_sde = [-2148, -2112,  -2119  ,-2153, -2161]
#loglikelhodd_ode = [-2148, -2112,  -2119  ,-2153, -2161]

noise_levels = [0.0, 0.2, 0.5, 1.0, 2.0]


bps_ode = [0.23, 0.2169, 0.207, 0.193, 0.186]
bps_sde = [0.247, 0.25547, 0.273, 0.248, 0.227]


bps_ode_std = [0.004, 0.005, 0.015, 0.01, 0.01]
bps_sde_std = [0.005, 0.005, 0.013, 0.007, 0.016]


fig = Figure(size = (800, 600))
ax = CairoMakie.Axis(fig[1, 1], xlabel = "noise level", ylabel = "bits per spike", limits = (nothing, (0.1, 0.35)),
    xticks = noise_levels, xgridvisible=false, ygridvisible=false)

CairoMakie.scatter!(ax, noise_levels, bps_ode, color = :orange, label = "latent neural ODE", markersize = 20)
CairoMakie.scatter!(ax, noise_levels, bps_sde, color = :dodgerblue2, label = "latent neural SDE", markersize = 20)

CairoMakie.errorbars!(noise_levels, bps_ode, bps_ode_std, color = :orange, whiskerwidth = 10)
CairoMakie.errorbars!(noise_levels, bps_sde, bps_sde_std, color = :dodgerblue2, whiskerwidth = 10)

lines!(ax, noise_levels, bps_ode, color = :orange, linestyle = :dash)
lines!(ax, noise_levels, bps_sde, color = :dodgerblue2, linestyle = :dash)

axislegend(backgroundcolor = :transparent)



f = fig 
save("wilson_cowan_bps.svg", f)