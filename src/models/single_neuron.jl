@with_kw struct FitzHughNagumo <: DynamicalSystem
    τ::Float32 = 1.0       # Time constant for membrane potential
    ε::Float32 = 0.08      # Time scale separation parameter
    a::Float32 = 0.7       # Parameter controlling the cubic nullcline
    b::Float32 = 0.8       # Recovery variable parameter
end

function Lux.initialparameters(rng::AbstractRNG, l::FitzHughNagumo)
    return (τ = l.τ, ε = l.ε, a = l.a, b = l.b)
end

Lux.initialstates(::AbstractRNG, ::FitzHughNagumo) = NamedTuple()

Lux.parameterlength(::FitzHughNagumo) = 4

Lux.statelength(::FitzHughNagumo) = 0

function (l::FitzHughNagumo)(x, u::Nothing, t, p, st)
    V = @view x[1:1, :]
    W = @view x[2:2, :]

    @unpack τ, ε, a, b = p

    dV = @. (V - (V^3)/3 - W) / τ
    dW = @. ε * (V + a - b * W)

    dx = vcat(dV, dW)

    return dx, st
end

# Optional: If you want to allow external current input
function (l::FitzHughNagumo)(x, u::AbstractArray, t, p, st)
    V = @view x[1:1, :]
    W = @view x[2:2, :]

    @unpack τ, ε, a, b = p

    I = u

    dV = @. (V - (V^3)/3 - W + I) / τ
    dW = @. ε * (V + a - b * W)

    dx = vcat(dV, dW)

    return dx, st
end


#############################################

@with_kw struct HodgkinHuxley <: DynamicalSystem
    g_Na::Float32 = 120.0     # Maximum conductance for sodium
    g_K::Float32 = 36.0       # Maximum conductance for potassium
    g_L::Float32 = 0.3        # Leak conductance
    E_Na::Float32 = 50.0      # Sodium reversal potential
    E_K::Float32 = -77.0      # Potassium reversal potential
    E_L::Float32 = -54.387    # Leak reversal potential
    C_m::Float32 = 1.0        # Membrane capacitance
end

function Lux.initialparameters(rng::AbstractRNG, l::HodgkinHuxley)
    return (g_Na = l.g_Na, g_K = l.g_K, g_L = l.g_L, E_Na = l.E_Na, E_K = l.E_K, E_L = l.E_L, C_m = l.C_m)
end

Lux.initialstates(::AbstractRNG, ::HodgkinHuxley) = NamedTuple()

Lux.parameterlength(::HodgkinHuxley) = 7

Lux.statelength(::HodgkinHuxley) = 0

# Define the auxiliary functions for the Hodgkin-Huxley model
α_m(V) = 0.1 * (V + 40.0) / (1.0 - exp(-0.1 * (V + 40.0)))
β_m(V) = 4.0 * exp(-0.0556 * (V + 65.0))
α_h(V) = 0.07 * exp(-0.05 * (V + 65.0))
β_h(V) = 1.0 / (1.0 + exp(-0.1 * (V + 35.0)))
α_n(V) = 0.01 * (V + 55.0) / (1.0 - exp(-0.1 * (V + 55.0)))
β_n(V) = 0.125 * exp(-0.0125 * (V + 65.0))

function (l::HodgkinHuxley)(x, u::Nothing, t, p, st)
    V = @view x[1:1, :]
    m = @view x[2:2, :]
    h = @view x[3:3, :]
    n = @view x[4:4, :]

    @unpack g_Na, g_K, g_L, E_Na, E_K, E_L, C_m = p
    
    I_Na = @. g_Na * m^3 * h * (V - E_Na)
    I_K = @. g_K * n^4 * (V - E_K)
    I_L = @. g_L * (V - E_L)

    dV = @. (0.0 - I_Na - I_K - I_L) / C_m  # No external current (I_ext = 0)
    dm = @. α_m(V) * (1.0 - m) - β_m(V) * m
    dh = @. α_h(V) * (1.0 - h) - β_h(V) * h
    dn = @. α_n(V) * (1.0 - n) - β_n(V) * n

    dx = vcat(dV, dm, dh, dn)

    return dx, st
end

# Optional: If you want to allow external current input
function (l::HodgkinHuxley)(x, u::AbstractArray, t, p, st)
    V = @view x[1:1, :]
    m = @view x[2:2, :]
    h = @view x[3:3, :]
    n = @view x[4:4, :]

    @unpack g_Na, g_K, g_L, E_Na, E_K, E_L, C_m = p
    
    I_ext = u

    I_Na = @. g_Na * m^3 * h * (V - E_Na)
    I_K = @. g_K * n^4 * (V - E_K)
    I_L = @. g_L * (V - E_L)

    dV = @. (I_ext - I_Na - I_K - I_L) / C_m
    dm = @. α_m(V) * (1.0 - m) - β_m(V) * m
    dh = @. α_h(V) * (1.0 - h) - β_h(V) * h
    dn = @. α_n(V) * (1.0 - n) - β_n(V) * n

    dx = vcat(dV, dm, dh, dn)

    return dx, st
end