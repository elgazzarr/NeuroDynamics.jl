
"""
    ODE(vector_field, solver; kwargs...)

Constructs an ODE model.

Arguments:

  - `vector_field`: The vector field of the ODE. 
  - `solver': The nummerical solver used to solve the ODE.
  - `kwargs`: Additional keyword arguments to pass to the solver.

"""
struct ODE{VF} <: UDE
    vector_field::VF
    solver
    kwargs
end

function ODE(vector_field, solver; kwargs...)
    return ODE(vector_field, solver, kwargs)
end

"""
    (de::ODE)(x::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, p::ComponentVector, st::NamedTuple)

The forward pass of the ODE.


Arguments:

  - `x`: The initial hidden state.
  - `u`: The control input.
  - `ts`: The time steps.
  - `p`: The parameters.
  - `st`: The state.

returns: 
    - The solution of the ODE.
    - The state of the model.

"""
function (de::ODE)(x::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, p::ComponentArray, st::NamedTuple)
    u_cont(t) = interp!(ts, u, t)
    dxdt(x, p, t) = dxdt_u(de.vector_field, x, u_cont(t), t, p, st)[1]
    ff = ODEFunction{false}(dxdt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, (ts[1], ts[end]), p)
    return solve(prob, de.solver; sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), de.kwargs...), st
end


"""
    sample_dynamics(de::ODE, x̂₀, u, ts, p, st, n_samples)

Samples trajectories from the ODE model.

Arguments:

  - `de`: The ODE model to sample from.
  - `x̂₀`: The initial hidden state.
  - `u`: Inputs for the input encoder. Can be `Nothing` or an array.
  - `ts`: Array of time points at which to sample the trajectories.
  - `p`: The parameters.
  - `st`: The state.
  - `n_samples`: The number of samples to generate.

returns: 
    - The sampled trajectories.
    - The state of the model.

"""
function sample_dynamics(de::ODE, x̂₀, u, ts, p, st, n_samples)
    u_cont(t) = interp!(ts, u, t)
    x₀ = sample_rp(x̂₀)
    dxdt(x, p, t) = dxdt_u(de.vector_field, x, u_cont(t), t, p, st)[1]
    ff = ODEFunction{false}(dxdt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x₀, (ts[1], ts[end]), p)

    function prob_func(prob, i, repeat)
        remake(prob, u0=sample_rp(x̂₀))
    end
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    ensemble_sol = solve(ensemble_prob, de.solver, EnsembleThreads(); trajectories=n_samples, de.kwargs...)
    x = permutedims(Array(ensemble_sol), (1, 3, 2, 4)) 
    return x
end


"""
    phaseplot(de::ODE, x₀_ranges, u, ts, p, st; kwargs...)

Plots the phase portrait of the ODE model.

Arguments:

  - `de`: The ODE model.
  - `x₀_ranges`: The initial condition ranges.
  - `u`: The control input.
  - `ts`: The time steps.
  - `p`: The parameters.
  - `st`: The state of the model.
  - `kwargs`: Additional keyword arguments for plotting.

"""
function phaseplot(de::ODE, x₀_ranges, u, ts, p, st; kwargs...)
    n_dims = length(x₀_ranges[1])
    u_cont(t) = interp!(ts, u, t)
    dt = ts[2] - ts[1]
    @assert n_dims in [2, 3] "The vector field must be 2D or 3D for plotting" 
    dxdt(x, p, t) = dxdt_u(de.vector_field, x, u_cont(t), t, p, st)[1]
    ff = ODEFunction{false}(dxdt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x₀_ranges, (0.0, dt), p)
    function prob_func(prob, i, repeat)
        x0 = collect(prob.u0[i])
        x0 = reshape(x0, size(x0)..., 1)
        remake(prob, u0 = x0)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    sim = solve(ensemble_prob, Euler(), EnsembleThreads(),  dt=dt, trajectories = length(x₀_ranges))
    data = Array(sim) .|> Float32
    data = data[:, 1, :, :]

    if n_dims == 2
        plot_phase_portrait_2d(data; kwargs...)
    else
        plot_phase_portrait_3d(data; kwargs...)
    end
    
end

################################################################################################################################################################

"""
    SDE(drift, drift_aug, diffusion, solver; kwargs...)

Constructs an SDE model.

Arguments:

  - `drift`: The drift of the SDE. 
  - `drift_aug`: The augmented drift of the SDE. Used only for training.
  - `diffusion`: The diffusion of the SDE.
  - `solver': The nummerical solver used to solve the SDE.
  - `kwargs`: Additional keyword arguments to pass to the solver.

"""
struct SDE{D, DA, DI} <: SUDE
    drift::D
    drift_aug::DA
    diffusion::DI
    solver
    kwargs
end

function SDE(drift, drift_aug, diffusion, solver; kwargs...)
    return SDE(drift, drift_aug, diffusion, solver, kwargs)
end


"""
    (de::SDE)(x::AbstractArray, u::AbstractArray, c::AbstractArray, ts::StepRangeLen, p::ComponentVector, st::NamedTuple)

The forward pass of the SDE.


Arguments:

  - `x`: The initial hidden state.
  - `u`: The control input.
  - `c`: The context.
  - `ts`: The time steps.
  - `p`: The parameters.
  - `st`: The state.

returns: 
    - The solution of the SDE.
    - The state of the model.
"""
function (de::SDE)(x::AbstractArray, u::Union{Nothing, AbstractArray}, c::AbstractArray, ts::AbstractArray, p::ComponentVector, st::NamedTuple)
    #TBD to fix the interpolation
    u_cont1(t) = interp!(ts, u, t)
    u_cont2(t) = interp!(ts, u, t)
    c_cont(t) = interp!(ts, c, t)

    function μ_posterior(x, p, t)
        c_t(t) = c_cont(t)
        u1(t) = u_cont1(t)
        xc = vcat(x, c_t(t), u1(t))
        return de.drift_aug(xc, p, st.drift_aug)[1]
    end

    function μ_prior(x, p, t)
        u2(t) = u_cont2(t)
        return dxdt_u(de.drift, x, u2(t), t, p, st.drift)[1]
    end

    function σ_shared(x, p, t)
        return de.diffusion(x, p, st.diffusion)[1]
    end

    function μ(x, p, t)
        x_ = x[1:end-1,:]
        f = μ_posterior(x_, p.drift_aug, t)
        h = μ_prior(x_, p.drift, t)
        g = σ_shared(x_, p.diffusion, t)
        s = (h.-f)./g 
        f_logqp = 0.5 .* sum(s.^2, dims = 1) 
        return vcat(f, f_logqp)
    end

    function σ(x, p, t)
        x_ = x[1:end-1,:]
        g = σ_shared(x_, p.diffusion, t) 
        g_logqp = CRC.@ignore_derivatives fill!(similar(x_, 1, size(x_)[end]), 0.0)
        return  vcat(g, g_logqp) 
    end
    
    ff = SDEFunction{false}(μ, σ)
    prob = SDEProblem{false}(ff, x, (ts[1], ts[end]), p)
    return solve(prob, de.solver; u0 = x, p, sensealg = TrackerAdjoint(), de.kwargs...), st
end



"""
    sample_dynamics(de::SDE, x̂₀, u, ts, p, st, n_samples)

Samples trajectories from the SDE model.

Arguments:

  - `de`: The SDE model to sample from.
  - `x̂₀`: The initial hidden state.
  - `u`: Inputs for the input encoder. Can be `Nothing` or an array.
  - `ts`: Array of time points at which to sample the trajectories.
  - `p`: The parameters.
  - `st`: The state.
  - `n_samples`: The number of samples to generate.

returns: 
    - The sampled trajectories.
    - The state of the model.

"""
function sample_dynamics(de::SDE, x̂₀, u, ts, p, st, n_samples)
    tspan = (ts[1], ts[end])
    u_cont(t) = interp!(ts, u, t)
    x₀ = sample_rp(x̂₀)
    μ(x, p, t) = dxdt_u(de.drift, x, u_cont(t), t, p.drift, st.drift)[1]
    σ(x, p, t) = de.diffusion(x, p.diffusion, st.diffusion)[1]

    ff = SDEFunction{false}(μ, σ, tgrad = basic_tgrad)
    prob = SDEProblem{false}(ff, x₀, tspan, p)

    function prob_func(prob, i, repeat)
        remake(prob, u0=sample_rp(x̂₀))
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    ensemble_sol = solve(ensemble_prob, de.solver, EnsembleThreads(); trajectories=n_samples, de.kwargs...)
    x = permutedims(Array(ensemble_sol), (1, 3, 2, 4)) 
    return x
end


#############################################################################################################################################
function dxdt_u(model::Lux.AbstractExplicitLayer, x, u, t, p, st)
    xu = vcat(x, u)
   return model(xu, p, st)
end

function dxdt_u(model::Lux.AbstractExplicitLayer, x, u::Nothing, t, p, st)
   return model(x, p, st)
end

# Specialize the helper for DynamicalSystemLayer
function dxdt_u(model::DynamicalSystem, x, u, t, p, st)
    return  model(x, u, t, p, st)
end

function dxdt_u(model::DynamicalSystem, x,  u::Nothing, t, p, st)
    return  model(x, nothing, t, p, st)
end