struct SlOscillators{F1, F2, F3, F4} <: DynamicalSystem
    N::Int                   # Number of oscillators
    M::Int                   # Number of inputs
    a::F1               # Initialization function for parameter a
    ω::F2           # Initialization function for parameter omega
    K::F3               # Initialization function for the global coupling strength K
    B::F4               # Initialization function for the input coupling matrix B
end

function SlOscillators(N::Int, M::Int; a=Lux.ones32, ω=Lux.rand32, K=Lux.rand32, B=Lux.zeros32)
    return SlOscillators{typeof(a), typeof(ω), typeof(K), typeof(B)}(N, M, a, ω, K, B)
end

function Lux.initialparameters(rng::AbstractRNG, l::SlOscillators)
    a = l.a(rng, l.N, 1)
    ω = l.ω(rng, l.N, 1)
    K = l.K(rng, l.N, l.N)
    B = l.B(rng, l.N, l.M)
    return (a=a, ω=ω, K=K, B=B)
end

Lux.initialstates(::AbstractRNG, ::SlOscillators) = NamedTuple()
Lux.parameterlength(l::SlOscillators) = 2 * l.N + l.N * l.N + l.N * l.M  # a, omega for each N and one K and one B
Lux.statelength(::SlOscillators) = 0


function (l::SlOscillators)(x, u::AbstractArray, t, p, st)
    N = l.N
    x_ = x[1:N, :]
    y_ = x[N+1:2N, :]
    a = p.a
    ω = p.ω
    K = p.K
    B = p.B
    # Compute dx and dy using matrix operations
    dx_ = a .* x_ .- ω .* y_ .- (x_ .^ 2 .+ y_ .^ 2) .* y_ .+ K  * x_ .+ B * u
    dy_ = a .* y_ .+ ω .* x_ .- (x_ .^ 2 .+ y_ .^ 2) .* y_ .+ K  * y_ .+ B * u
    # Concatenate dx and dy to form the output
    dx = vcat(dx_, dy_)

    return dx, st
end


function (l::SlOscillators)(x, u::Nothing, t, p, st)
    N = l.N
    x_ = x[1:N, :]
    y_ = x[N+1:2N, :]
    a = p.a
    ω = p.ω
    K = p.K

    # Compute dx and dy using matrix operations
    dx_ = a .* x_ .- ω .* y_ .- (x_ .^ 2 .+ y_ .^ 2) .* y_ .+ K * x_ 
    dy_ = a .* y_ .+ ω .* x_ .- (x_ .^ 2 .+ y_ .^ 2) .* y_ .+ K * y_ 

    # Concatenate dx and dy to form the output
    dx = vcat(dx_, dy_)

    return dx, st
end


struct BistableOscillators{F1, F2, F3} <: DynamicalSystem
    N::Int                   # Number of oscillators
    init_sigma::F1           # Initialization function for parameter σ
    init_omega::F2           # Initialization function for parameter ω
    init_K::F3               # Initialization function for the global coupling strength K
end

function BistableOscillators(N::Int; init_sigma=Lux.rand32, init_omega=Lux.rand32, init_K=Lux.rand32)
    return BistableOscillators{typeof(init_sigma), typeof(init_omega), typeof(init_K)}(N, init_sigma, init_omega, init_K)
end

function Lux.initialparameters(rng::AbstractRNG, l::BistableOscillators)
    σ = l.init_sigma(rng, l.N, 1) 
    ω = l.init_omega(rng, l.N, 1)
    K = l.init_K(rng, l.N, l.N)
    return (σ=σ, ω=ω, K=K)
end

Lux.initialstates(::AbstractRNG, ::BistableOscillators) = NamedTuple()
Lux.parameterlength(l::BistableOscillators) = 2 * l.N + l.N * l.N 

Lux.statelength(::BistableOscillators) = 0


function (l::BistableOscillators)(x, u::Nothing, t, p, st)
    N = l.N
    x_ = x[1:N, :]
    y_ = x[N+1:2N, :]
    
    σ = p.σ
    ω = p.ω
    K = p.K

    # Compute dx and dy using matrix operations
    dx = (-ω.*y_ + x_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2) + K * x_) .+ Float32(1e-4)
    dy = (ω.*x_ + y_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2)  + K * y_).+ Float32(1e-4)

    # Concatenate dx and dy to form the output
    dxy = vcat(dx, dy)

    return dxy, st
end

function (l::BistableOscillators)(x, u::AbstractArray, t, p, st)
    N = l.N
    x_ = x[1:N, :]
    y_ = x[N+1:2N, :]
    
    σ = p.σ
    ω = p.ω
    K = p.K

    # Compute dx and dy using matrix operations
    dx = (-ω.*y_ + x_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2) + K * x_ +  u) .+ Float32(1e-4)
    dy = (ω.*x_ + y_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2)  + K * y_ +  u).+ Float32(1e-4)

    # Concatenate dx and dy to form the output
    dxy = vcat(dx, dy)

    return dxy, st
end




struct HarmonicOscillators{F1, F2, F3} <: DynamicalSystem
    N::Int                # Number of oscillators
    init_omega::F1        # Initialization function for parameter ω
    init_gamma::F2        # Initialization function for the damping coefficient γ
    init_B::F3            # Initialization function for the external input coupling matrix B
end

function HarmonicOscillators(N::Int; init_omega=Lux.rand32, init_gamma=Lux.ones32, init_B=Lux.rand32)
    return HarmonicOscillators{typeof(init_omega), typeof(init_gamma), typeof(init_B)}(N, init_omega, init_gamma, init_B)
end

function Lux.initialparameters(rng::AbstractRNG, l::HarmonicOscillators)
    ω = l.init_omega(rng, l.N, 1)
    γ = l.init_gamma(rng, l.N, l.N)
    B = l.init_B(rng, l.N, l.N)
    return (ω=ω, γ=γ, B=B)
end

Lux.initialstates(::AbstractRNG, ::HarmonicOscillators) = NamedTuple()
Lux.parameterlength(m::HarmonicOscillators) = length(m.init_omega) + size(m.init_gamma, 1) * size(m.init_gamma, 2) + size(m.init_B, 1) * size(m.init_B, 2)
Lux.statelength(::HarmonicOscillators) = 0

function (m::HarmonicOscillators)(state, u::AbstractArray, t, p, st)
    N, ω, γ, B = p.ω, p.γ, p.B
    x, v = state[1:N], state[N+1:2N]
    dx = v
    dv = -γ * v .- ω.^2 .* x .+ B * u
    return vcat(dx, dv), st
end


function (m::HarmonicOscillators)(state, u::Nothing, t, p, st)
    N, ω, γ, B = p.ω, p.γ, p.B
    x, v = state[1:N], state[N+1:2N]
    dx = v
    dv = -γ * v .- ω.^2 .* x
    return vcat(dx, dv), st
end
