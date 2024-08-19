"""

Wilson-Cowan Model of Neural Populations 

The classic Wilson-Cowan model is a system of ordinary differential equations that describes the dynamics of excitatory and inhibitory neural populations.
"""
struct WilsonCowan <: DynamicalSystem
    Ne::Int                 # Number of excitatory neurons
    Ni::Int                 # Number of inhibitory neurons
    tauE                    # Time constant for excitatory neurons
    tauI                    # Time constant for inhibitory neurons
    aE                      # Gain term for excitatory neurons
    aI                      # Gain term for inhibitory neurons
    thetaE                  # Threshold term for excitatory neurons
    thetaI                  # Threshold term for inhibitory neurons
    wEE                     # Weight for E to E connection
    wEI                     # Weight for I to E connection
    wIE                     # Weight for E to I connection
    wII                     # Weight for I to I connection
end

# Constructor for the Wilson-Cowan model
function WilsonCowan(Ne::Int, Ni::Int; 
    tauE=Lux.ones32, tauI=Lux.ones32, aE=Lux.rand32, aI=Lux.rand32, 
    thetaE=Lux.rand32, thetaI=Lux.rand32, wEE=Lux.rand32, wEI=Lux.rand32, 
    wIE=Lux.rand32, wII=Lux.rand32)
    return WilsonCowan(
        Ne, Ni, tauE, tauI, aE, aI, thetaE, thetaI, wEE, wEI, wIE, wII)
end

# Initialize parameters for the Wilson-Cowan model
function Lux.initialparameters(rng::AbstractRNG, l::WilsonCowan)
    tauE = l.tauE(rng, l.Ne, 1)
    tauI = l.tauI(rng, l.Ni, 1)
    aE = l.aE(rng, l.Ne, 1)
    aI = l.aI(rng, l.Ni, 1)
    thetaE = l.thetaE(rng, l.Ne, 1)
    thetaI = l.thetaI(rng, l.Ni, 1)
    wEE = l.wEE(rng, l.Ne, l.Ne)
    wEI = l.wEI(rng, l.Ne, l.Ni)
    wIE = l.wIE(rng, l.Ni, l.Ne)
    wII = l.wII(rng, l.Ni, l.Ni)
    return (tauE=tauE, tauI=tauI, aE=aE, aI=aI, thetaE=thetaE, thetaI=thetaI, wEE=wEE, wEI=wEI, wIE=wIE, wII=wII)
end

Lux.initialstates(::AbstractRNG, ::WilsonCowan) = NamedTuple()
Lux.parameterlength(l::WilsonCowan) = l.Ne + l.Ni + l.Ne + l.Ni + l.Ne + l.Ni + l.Ne*l.Ne + l.Ne*l.Ni + l.Ni*l.Ne + l.Ni*l.Ni + l.Ne + l.Ni
Lux.statelength(::WilsonCowan) = 0

# Define the dynamical equations for the Wilson-Cowan model
function (l::WilsonCowan)(x, u::AbstractArray, t, p, st)
    Ne = l.Ne
    Ni = l.Ni

    rE = x[1:Ne, :]
    rI = x[Ne+1:Ne+Ni, :]
    
    tauE = p.tauE
    tauI = p.tauI
    aE = p.aE
    aI = p.aI
    thetaE = p.thetaE
    thetaI = p.thetaI
    wEE = p.wEE
    wEI = p.wEI
    wIE = p.wIE
    wII = p.wII

    Ie = u[1:Ne, :]
    Ii = u[Ne+1:Ne+Ni, :]
    
    FE = x -> 1 ./ (1 .+ exp.(-aE .* (x .- thetaE))) .- 1 ./ (1 .+ exp.(aE .* thetaE))
    FI = x -> 1 ./ (1 .+ exp.(-aI .* (x .- thetaI))) .- 1 ./ (1 .+ exp.(aI .* thetaI))
    
    dE = (-rE .+ FE(wEE * rE .- wEI * rI .+ Ie)) ./ tauE
    dI = (-rI .+ FI(wIE * rE .- wII * rI .+ Ii)) ./ tauI
    
    dx = vcat(dE, dI)
    
    return dx, st
end

function (l::WilsonCowan)(x,  u::Nothing, t, p, st)
    Ne = l.Ne
    Ni = l.Ni

    rE = x[1:Ne, :]
    rI = x[Ne+1:Ne+Ni, :]
    
    tauE = p.tauE
    tauI = p.tauI
    aE = p.aE
    aI = p.aI
    thetaE = p.thetaE
    thetaI = p.thetaI
    wEE = p.wEE
    wEI = p.wEI
    wIE = p.wIE
    wII = p.wII
    
    FE = x -> 1 ./ (1 .+ exp.(-aE .* (x .- thetaE))) .- 1 ./ (1 .+ exp.(aE .* thetaE))
    FI = x -> 1 ./ (1 .+ exp.(-aI .* (x .- thetaI))) .- 1 ./ (1 .+ exp.(aI .* thetaI))
    
    dE = (-rE .+ FE(wEE * rE .- wEI * rI)) ./ tauE
    dI = (-rI .+ FI(wIE * rE .- wII * rI)) ./ tauI
    
    dx = vcat(dE, dI)
    
    return dx, st
end



#######################################################################

"""

Modern Wilson-Cowan Model of Neural Populations 

The modern Wilson-Cowan model is a system of ordinary differential equations that describes the dynamics of excitatory and inhibitory neural populations with external inputs.

"""
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


function (m::ModernWilsonCowan)(x, u::AbstractArray, t, p, st) 
    dx = (-x + p.J * tanh.(x) + p.B * u .+ p.b)./(p.τ .+ 0.001f0)
    return dx, st
end

function (m::ModernWilsonCowan)(x, u::Nothing, t, p, st)
    dx = (-x + p.J * tanh.(x) .+ p.b)./(p.τ .+ 0.001f0)
    return dx, st
end



##########################################


struct JansenRit{F1, F2, F3, F4, F5, F6, F7} <: DynamicalSystem
    N::Int              # Number of populations (fixed at 6)
    M::Int              # Input size
    α::F1               # Parameter α
    β::F2               # Parameter β
    C1::F3              # Coupling parameter C1
    C2::F4              # Coupling parameter C2
    C3::F5              # Coupling parameter C3
    C4::F6              # Coupling parameter C4
    C5::F7              # Coupling parameter C5
end

function JansenRit(M::Int; α=5.0f0, β=20.0f0, C1=135.0f0, C2=108.0f0, C3=33.75f0, C4=33.75f0, C5=108.0f0)
    N = 6
    return JansenRit(N, M, [α], [β], [C1], [C2], [C3], [C4], [C5])
end

function Lux.initialparameters(rng::AbstractRNG, j::JansenRit)
    α = j.α
    β = j.β
    C1 = j.C1
    C2 = j.C2
    C3 = j.C3
    C4 = j.C4
    C5 = j.C5
    return (α=α, β=β, C1=C1, C2=C2, C3=C3, C4=C4, C5=C5)
end

Lux.initialstates(::AbstractRNG, ::JansenRit) = NamedTuple()

Lux.parameterlength(j::JansenRit) = 7  # α, β, C1, C2, C3, C4, C5 each in 1x1 array
Lux.statelength(::JansenRit) = 6

function (j::JansenRit)(x, u, t, p, st)
    @unpack α, β, C1, C2, C3, C4, C5 = p
        
    y = reshape(x[1:3], 3)
    dy = reshape(x[4:6], 3)
    
    ddy0 = α^2 .* (C2 .* tanh.(y[2] .- y[3]) .+ C5 .* u[1]) .- 2 .* α .* dy[1] .- α^2 .* y[1]
    ddy1 = β^2 .* (C4 .* tanh.(C3 .* y[1])) .- 2 .* β .* dy[2] .- β^2 .* y[2]
    ddy2 = β^2 .* (C1 .* tanh.(C3 .* y[1])) .- 2 .* β .* dy[3] .- β^2 .* y[3]
    
    dx = vcat(dy, [ddy0, ddy1, ddy2])
    
    return dx, st
end