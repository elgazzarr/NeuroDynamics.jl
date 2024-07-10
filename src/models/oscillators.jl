struct BistableOscillators{F1, F2, F3, F4} <: DynamicalSystem
    N::Int             # Number of oscillators
    M::Int            # Number of inputs
    σ::F1              # Bifurcation parameter σ
    ω::F2              # Frequency parameter ω
    K::F3              # Coupling matrix K 
    B::F4                  # Input coupling matrix B
end

function BistableOscillators(N::Int; M = 0, init_sigma=Lux.rand32, init_omega=Lux.rand32, init_K=Lux.rand32, init_B=Lux.zeros32)
    return BistableOscillators{typeof(init_sigma), typeof(init_omega), typeof(init_K), typeof(init_B)}(N, M, init_sigma, init_omega, init_K, init_B)
end

function Lux.initialparameters(rng::AbstractRNG, l::BistableOscillators)
    σ = l.σ(rng, l.N, 1) 
    ω = l.ω(rng, l.N, 1)
    K = l.K(rng, l.N, l.N)
    B = l.B(rng, l.N, l.M)
    return (σ=σ, ω=ω, K=K, B=B)
end

Lux.initialstates(::AbstractRNG, ::BistableOscillators) = NamedTuple()
Lux.parameterlength(l::BistableOscillators) = 2 * l.N + l.N * l.N 

Lux.statelength(::BistableOscillators) = 0


function (l::BistableOscillators)(x, u::Nothing, t, p, st)
    N = l.N
    x_ = x[1:N, :]
    y_ = x[N+1:2N, :]

    @unpack σ, ω, K, B = p

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
    
    @unpack σ, ω, K, B = p

    # Compute dx and dy using matrix operations
    dx = (-ω.*y_ + x_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2) + K * x_ +  B*u) .+ Float32(1e-4)
    dy = (ω.*x_ + y_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2)  + K * y_ +  B*u) .+ Float32(1e-4)

    # Concatenate dx and dy to form the output
    dxy = vcat(dx, dy)

    return dxy, st
end


###################################################################################
"""
    HarmonicOscillators

A struct that defines a system of harmonic oscillators.

# Arguments
- `N' : Number of oscillators
- `M' : Number of inputs
- `ω' : Frequency parameter ω
- `γ' : Damping coefficient γ
- `K' : Coupling matrix K
- `B' : External input coupling matrix B

"""
@with_kw struct HarmonicOscillators{F1, F2, F3, F4} <: DynamicalSystem
    N::Int                # Number of oscillators
    M::Int               # Number of inputs
    ω::F1                 # Frequency parameter ω
    γ::F2                 # Damping coefficient γ
    K::F3                 # Coupling matrix K
    B::F4                 # External input coupling matrix B
end

function HarmonicOscillators(N::Int; M=0, ω=Lux.rand32, γ=Lux.ones32, K=Lux.rand32, B=Lux.rand32)
    return HarmonicOscillators{typeof(ω), typeof(γ), typeof(K), typeof(B)}(N, M, ω, γ, K, B)
end

function Lux.initialparameters(rng::AbstractRNG, l::HarmonicOscillators)
    ω = l.ω(rng, l.N, 1)
    γ = l.γ(rng, l.N, 1)
    K = l.K(rng, l.N, l.N)
    B = l.B(rng, l.N, l.M)
    return (ω=ω, γ=γ, K=K, B=B)
end

Lux.initialstates(::AbstractRNG, ::HarmonicOscillators) = NamedTuple()

Lux.parameterlength(m::HarmonicOscillators) = m.N + m.N + m.N * m.N + m.N * m.M

Lux.statelength(::HarmonicOscillators) = 0

"""
    (m::HarmonicOscillators)(x, u, t, p, st)

The forward pass of the HarmonicOscillators.


"""
function (m::HarmonicOscillators)(x, u::AbstractArray, t, p, st)
    N = m.N
    x_ = @view x[1:N, :]
    v_ = @view x[N+1:2N, :]

    @unpack ω, γ, K, B = p

    dx = @. v_
    dv = @. -γ * v_ - ω^2 * x_ + K * x_ + B * u

    return vcat(dx, dv), st
end

function (m::HarmonicOscillators)(x, nothing, t, p, st)
    N = m.N
    x_ = @view x[1:N, :]
    v_ = @view x[N+1:2N, :]

    @unpack ω, γ, K = p

    dx = @. v_
    dv = @. -γ * v_ - ω^2 * x_ + K * x_

    return vcat(dx, dv), st
end



