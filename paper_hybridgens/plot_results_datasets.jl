using CairoMakie


colors = Makie.wong_colors()

datasets = ["mc_maze (Primary Motor Cortex)", "area2_bump (Area2)"]


dynamics_models = ["SlOscillators", "WilsonCowan", "JensenRit", "MLP"]


groups = ["sde", "ode"]


tbl = (cat = [1, 1, 2, 2, 3,3, 4, 4],
       val_bps = [0.094, 0.135,  0.092, 0.128, 0.0907, 0.124, 0.1008, 0.130],
       err = abs.(randn(8) * 0.01),
       grp = [2, 1, 2, 1, 2, 1, 2, 1],
       )



fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1]; xlabel = "models", ylabel = "bits per spike (bps)", title = "mc_maze", xticks = (1:4, dynamics_models))

barplot!(ax, tbl.cat, tbl.val_bps,
        dodge = tbl.grp,
        color = colors[tbl.grp])


# Legend
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(groups)]

Legend(fig[1,2], elements, groups)

fig
save("mc_maze.pdf", fig)


