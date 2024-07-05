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
    dx = (-x + p.J * tanh.(x) + p.B * u .+ p.b)./(p.τ .+ 1e-3)
    return dx, st
end

function (m::ModernWilsonCowan)(x, ::Nothing, t, p, st)
    dx = (-x + p.J * tanh.(x) .+ p.b)./(p.τ .+ 1e-3)
    return dx, st
end