struct SlOscillators{F1, F2, F3, F4} <: DynamicalSystem
    N::Int             # Number of oscillators
    M::Int             # Number of inputs
    σ::F1              # Bifurcation parameter σ
    ω::F2              # Frequency parameter ω
    K::F3              # Coupling matrix K 
    B::F4              # Input coupling matrix B
end

function SlOscillators(N::Int; M = 0, init_sigma=Lux.rand32, init_omega=Lux.rand32, init_K=Lux.rand32, init_B=Lux.zeros32)
    return SlOscillators{typeof(init_sigma), typeof(init_omega), typeof(init_K), typeof(init_B)}(N, M, init_sigma, init_omega, init_K, init_B)
end

function Lux.initialparameters(rng::AbstractRNG, l::SlOscillators)
    σ = l.σ(rng, l.N, 1) 
    ω = l.ω(rng, l.N, 1)
    K = l.K(rng, l.N, l.N)
    B = l.B(rng, l.N, l.M)
    return (σ=σ, ω=ω, K=K, B=B)
end

Lux.initialstates(::AbstractRNG, ::SlOscillators) = NamedTuple()
Lux.parameterlength(l::SlOscillators) = 2 * l.N + l.N ^2 + l.N * l.M 
Lux.statelength(::SlOscillators) = 0


function (l::SlOscillators)(x, u, t, p, st)
    N = l.N
    x_ = x[1:N, :]
    y_ = x[N+1:2N, :]

    @unpack σ, ω, K, B = p

    dx = (-ω.*y_ + x_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2) + K * x_ + B * u) 
    dy = (ω.*x_ + y_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2)  + K * y_ + B * u)

    dxy = vcat(dx, dy)

    return dxy, st
end


###############################################################

struct ModernWilsonCowan <: DynamicalSystem
    N::Int              # State size
    M::Int              # Input size
    τ                   # Time constant vector τ
    J                   # Synaptic coupling matrix J
    B                   # Input coupling matrix B  
    b                   # Input bias vector b
end

function ModernWilsonCowan(N::Int, M::Int; τ=Lux.ones32, J=Lux.glorot_uniform, B=Lux.glorot_uniform, b=Lux.ones32)
    return ModernWilsonCowan(N, M, τ, J, B, b)
end

function Lux.initialparameters(rng::AbstractRNG, m::ModernWilsonCowan)
    τ = m.τ(rng, m.N, 1)
    J = m.J(rng, m.N, m.N)
    B = m.B(rng, m.N, m.M)
    b = m.b(rng, m.N, 1)
    return (τ=τ, J=J, B=B, b=b)
end 

Lux.initialstates(::AbstractRNG, ::ModernWilsonCowan) = NamedTuple()
Lux.parameterlength(m::ModernWilsonCowan) =  m.N^2 + m.N + m.N*m.M + m.N
Lux.statelength(::ModernWilsonCowan) = 0


function (m::ModernWilsonCowan)(x, u, t, p, st) 
    dx = (-x + p.J * tanh.(x) + p.B * u .+ p.b)./(p.τ .+ 0.01f0)
    return dx, st
end


######################################################################
