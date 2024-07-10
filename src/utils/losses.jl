"""
    kl_normal(μ, σ²)

Compute the KL divergence between a normal distribution and a standard normal distribution.

Arguments:

  - `μ`: Mean of the normal distribution.
  - `σ²`: Variance of the normal distribution.

returns: 

    - The KL divergence.

"""
function kl_normal(μ, σ²)
    kl = 0.5 * sum(σ² .+ μ .^ 2 .- 1 .- log.(σ²))
    return kl
end

""" 
    poisson_loglikelihood(λ, y)

Compute the log-likelihood of a Poisson distribution.

Arguments:

  - `λ`: The rate of the Poisson distribution.
  - `y`: The observed spikes.

returns: 

    - The log-likelihood.

"""
function poisson_loglikelihood(λ, y) 
    @assert size(λ) == size(y) "poisson_loglikelihood: Rates and spikes should be of the same shape"
    @assert !any(isnan.(λ)) "poisson_loglikelihood: NaN rate predictions found"
    @assert all(λ .>= 0) "poisson_loglikelihood: Negative rate predictions found"
    λ = λ .+ 1e-6
    ll = sum(y .* log.(λ) .- λ .- lgamma.(y .+ 1))
    return ll 
end


"""
    normal_loglikelihood(μ, σ², y)

Compute the log-likelihood of a normal distribution.

Arguments:

  - `μ`: Mean of the normal distribution.
  - `σ²`: Variance of the normal distribution.
  - `y`: The observed values.

returns: 

    - The log-likelihood.

"""
function normal_loglikelihood(μ, σ², y)
    ll = -0.5 * sum(log.(2π * σ²) + ((y - μ).^2 ./ σ²))
    return ll
end


"""
    mse(ŷ, y)

Compute the mean squared error.

Arguments:

  - `ŷ`: Predicted values.
  - `y`: Observed values.

returns: 

    - The mean squared error.

"""
function mse(ŷ, y)
    return sum(abs, ŷ .- y)
end

"""
    bits_per_spike(rates, spikes)

Compute the bits per spike by comparing the Poisson log-likelihood of the rates with the Poisson log-likelihood of the mean spikes. 

Arguments:

  - `rates`: The predicted rates.
  - `spikes`: The observed spikes.

returns: 

    - The bits per spike.

"""
function bits_per_spike(rates, spikes)
    @assert size(rates) == size(spikes) "Rates and spikes must have the same shape"

    rates_ll = poisson_loglikelihood(rates, spikes)
    mean_spikes = mean(spikes, dims=(2, 3))
    null_rates = repeat(mean_spikes, 1, size(spikes, 2), size(spikes, 3)) 
    null_ll = poisson_loglikelihood(null_rates, spikes)
    spike_sum = sum(spikes)
    bps = (rates_ll/log(2) - null_ll/log(2)) / spike_sum
    return bps
end

"""
    frange_cycle_linear(n_iter, start, stop, n_cycle, ratio)

Generate a linear schedule with cycles.

Arguments:

  - `n_iter`: Number of iterations.
  - `start`: Start value.
  - `stop`: Stop value.
  - `n_cycle`: Number of cycles.
  - `ratio`: Ratio of the linear schedule.

returns: 

    - The linear schedule.

"""
function frange_cycle_linear(n_iter, start::T=0.0f0, stop::T=1.0f0,  n_cycle=4, ratio=0.5) where T
    L = ones(n_iter) * stop
    period = n_iter/n_cycle
    step = T((stop-start)/(period*ratio)) # linear schedule

    for c in 0:n_cycle-1
        v, i = start, 1
        while (v ≤ stop) & (Int(round(i+c*period)) < n_iter)
            L[Int(round(i+c*period))] = v
            v += step
            i += 1
        end
    end
    return T.(L)
end
