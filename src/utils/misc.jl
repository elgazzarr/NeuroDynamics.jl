"""
    sample_rp(x::Tuple)

Samples from a MultiVariate Normal distribution using the reparameterization trick.

Arguments:

  - `x`: Tuple of the mean and squared variance of a MultiVariate Normal distribution.

returns: 

    - The sampled value.
"""
sample_rp(x::Tuple) = x[1] + randn(size(x[1])...) .* sqrt.(x[2])
sample_rp(x::AbstractArray) = x
sample_rp(x::AbstractFloat) = x



"""
    interp!(ts, cs, time_point)

Interpolates the control signal at a given time point.

Arguments:

  - `ts`: Array of time points.
  - `cs`: Array of control signals.
  - `time_point`: The time point at which to interpolate the control signal.

returns: 

    - The interpolated control signal.

"""
function interp!(ts, cs::AbstractArray, time_point)
    # Use a generator to create interpolations and yield results
    return Zygote.@ignore [LinearInterpolation(ts, cs[i, :, j])(time_point) for i in 1:size(cs, 1), j in 1:size(cs, 3)]
end

function interp!(ts, cs::Nothing, time_point)
    return nothing
end

dropmean(A; dims=:) = dropdims(mean(A; dims=dims); dims=dims)


basic_tgrad(u, p, t) = zero(u)


# Custom vcat function for handling `nothing` values
function Base.vcat(a::AbstractArray, b::Nothing, c::AbstractArray)
    return vcat(a, c)
end

function Base.vcat(a::Nothing, b::AbstractArray, c::AbstractArray)
    return vcat(b, c)
end

function Base.vcat(a::AbstractArray, b::AbstractArray, c::Nothing)
    return vcat(a, b)
end

function Base.vcat(a::AbstractArray, b::Nothing)
    return a
end

function Base.vcat(a::Nothing, b::AbstractArray)
    return b
end
