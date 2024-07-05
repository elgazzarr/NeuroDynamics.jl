
function plot_samples(ŷ, sample_n=1; kwargs...)
    ŷₘ = selectdim(dropmean(ŷ, dims=4), 3, sample_n)
    ŷₛ = selectdim(dropmean(std(ŷ, dims=4), dims=4), 3, sample_n)
    plot(transpose(ŷₘ), ribbon=transpose(ŷₛ), legend=false, grid=false, alpha=0.2, label=["prediction" nothing])
end


function plot_preds(ys, ŷs, sample_n=1; kwargs...)
    plot_samples(ŷs, sample_n)
    scatter!(transpose(ys[:,:,sample_n]), label=["ground truth" nothing] , color="green", grid=false, markersize=2)
end

function plot_ci(y,  ŷₘ, ŷₛ, ci_factor=1.96, sample_n=1)
    up = ŷₘ .+ ci_factor .* sqrt.(ŷₛ)
    lb = ŷₘ .- ci_factor .* sqrt.(ŷₛ)
    up = selectdim(dropmean(up, dims=4), 3, sample_n)
    lb = selectdim(dropmean(lb, dims=4), 3, sample_n)
    ŷ_model = selectdim(dropmean(ŷₘ, dims=4), 3, sample_n)
    y = selectdim(y, 3, sample_n)
    scatter(transpose(y), label=["ground truth" nothing], color="green", grid=false, markersize=2)
    plot!(transpose(ŷ_model), color="red", grid=false, alpha=0.5, label=["mean prediction" nothing])
    plot!(transpose(up), fillrange=transpose(lb), fillalpha=0.3, label=["95% CI" nothing], lw=0, color="red", grid=false)
end

function plot_phase_portrait_2d(solution::Array{T, 3}) where T
    N, T_, B = size(solution)
    @assert N == 2 "State size must be 2 for 2D phase portrait"

    # Extract initial and next state points
    X0 = solution[:, 1, :]  # Initial state [N, B]
    X1 = solution[:, 2, :]  # Next state [N, B]

    # Compute direction vectors
    us = X1[1, :] .- X0[1, :]
    vs = X1[2, :] .- X0[2, :]

    # Initial positions
    xs = X0[1, :]
    ys = X0[2, :]

    # Compute strength for color mapping
    strength = sqrt.(us .^ 2 .+ vs .^ 2)
    cmap = :gnuplot

    # Create figure and axis
    fig = Figure(resolution = (600, 400))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", aspect = DataAspect())

    # Plot arrows
    arrows!(ax, xs, ys, us, vs, arrowsize = 8, lengthscale = 0.1,
        arrowcolor = strength, linecolor = strength, colormap = cmap)

    # Add colorbar
    Colorbar(fig[1, 2], limits = (minimum(strength), maximum(strength)),
        nsteps = 100, colormap = cmap, ticksize = 15, width = 15, tickalign = 1)

    # Set axis limits and aspect ratio
    limits!(ax, minimum(xs) - 1, maximum(xs) + 1, minimum(ys) - 1, maximum(ys) + 1)
    colsize!(fig.layout, 1, Aspect(1, 1.0))

    # Display figure
    display(fig)
end


function plot_phase_portrait_3d(solution::Array{T, 3}) where T
    N, T_, B = size(solution)
    @assert N == 3 "State size must be 3 for 3D phase portrait"

    # Extract initial and next state points
    X0 = solution[:, 1, :]  # Initial state [N, B]
    X1 = solution[:, 2, :]  # Next state [N, B]

    # Compute direction vectors
    us = X1[1, :] .- X0[1, :]
    vs = X1[2, :] .- X0[2, :]
    ws = X1[3, :] .- X0[3, :]

    # Initial positions
    xs = X0[1, :]
    ys = X0[2, :]
    zs = X0[3, :]

    # Compute strength for color mapping
    strength = sqrt.(us .^ 2 .+ vs .^ 2 .+ ws .^ 2)
    cmap = :gnuplot

    # Create figure and axis
    fig = Figure(resolution = (800, 600))
    ax = Axis3(fig[1, 1], xlabel = "x", ylabel = "y", zlabel = "z", aspect = DataAspect())

    # Plot arrows
    for i in 1:B
        arrows!(ax, [xs[i]], [ys[i]], [zs[i]], [us[i]], [vs[i]], [ws[i]],
            arrowsize = 0.2, lengthscale = 0.1, arrowcolor = strength[i], colormap = cmap)
    end

    # Add colorbar
    Colorbar(fig[1, 2], limits = (minimum(strength), maximum(strength)),
        nsteps = 100, colormap = cmap, ticksize = 15, width = 15, tickalign = 1)

    # Set axis limits and aspect ratio
    limits!(ax, minimum(xs) - 1, maximum(xs) + 1, minimum(ys) - 1, maximum(ys) + 1, minimum(zs) - 1, maximum(zs) + 1)
    colsize!(fig.layout, 1, Aspect(1, 1.0))

    # Display figure
    display(fig)
end


function animate_sol(sol, var_index, title, xlabel, ylabel, color)
    animation = @animate for i in 1:length(sol.t)
        plot(sol.t[1:i], sol[var_index, 1:i],
             title = title,
             xlabel = xlabel,
             ylabel = ylabel,
             label = nothing,
             lw = 3,
             color = color,
             ylim = (minimum(sol[var_index,:]) - 1.0, maximum(sol[var_index,:]) + 1.0))
        # Add scatter point for the most recent point
        scatter!([sol.t[i]], [sol[var_index, i]], color=:red, markersize=6, legend=:topright)
    end
    return animation
end


function animate_timeseries(x; kwargs...)
    base_plot = plot(xlabel="Time", legend=false, xlims=(0, size(x, 2)), grid=false, kwargs...)
    @animate for j in 1:size(x, 2)
        p = deepcopy(base_plot) # Start with a fresh copy of the base plot for each frame
        plot!(p, transpose(x[1:end, 1:j, 2]), legend=false, lw=2)
    end
end


function animate_oscillators(z)
    N = Int(size(z, 1) / 2)
    x = z[1:N, :]
    y = z[N+1:end, :]
    base_plot = plot(xlabel="Re(z)", ylabel="Im(z)", xlims=(-10, 10), ylims=(-10, 10), legend=false)
    @animate for j in 1:size(x, 2)
        p = deepcopy(base_plot) 
        plot!(p, transpose(x[1:N, 1:j]), transpose(y[1:N, 1:j]), legend=false, grid=false, lw=1.5, alpha=0.8)
        scatter!(p, x[1:N, j], y[1:N, j], legend=false)
    end
end



"""function animate_spikes(ŷ_out)
    base_plot = plot(xlabel="Time", ylabel="Spike count", legend=false, xlims=(0, size(ŷ_out, 2)), ylims=(0, 3), grid=false, xticks=false)
    dist = Poisson.(ŷ_out)
    y_poiss = rand.(dist)
    @gif for j in 1:size(ŷ_out, 2)
        p = deepcopy(base_plot) # Start with a fresh copy of the base plot for each frame
        plot!(p, transpose(y_poiss[1:end, 1:j, 1]), legend=false)
    end
end"""