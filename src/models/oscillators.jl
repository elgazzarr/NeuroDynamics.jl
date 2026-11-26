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

    dx = v_
    # K * x_ and B * u are matrix products; everything else is element-wise.
    dv = @. -γ * v_ - ω^2 * x_ + $(K * x_) + $(B * u)

    return vcat(dx, dv), st
end

function (m::HarmonicOscillators)(x, u::Nothing, t, p, st)
    N = m.N
    x_ = @view x[1:N, :]
    v_ = @view x[N+1:2N, :]

    @unpack ω, γ, K = p

    dx = v_
    dv = @. -γ * v_ - ω^2 * x_ + $(K * x_)

    return vcat(dx, dv), st
end





###################################################################################
"""
    CoupledOscillators(N::Int, M::Int; hidden_dim=16, init_alpha=Lux.rand32, init_omega=Lux.rand32)

Drift of the coupled-oscillator latent SDE/ODE (CO-SDE / CO-ODE) from
ElGazzar & van Gerven, *Generative Modeling of Neural Dynamics via Latent SDEs*
(arXiv:2412.12112), Eqs. (8)–(9).

The latent state of `N` oscillators is `x = vcat(a, b)` with length `2N`, where `a` holds the real
parts and `b` the imaginary parts. Batches are stored as columns, the same layout as
`BistableOscillators`. Each oscillator is a Hopf normal form whose growth rate is modulated
by a scalar, input-dependent coupling κ(u):

```math
\\begin{aligned}
\\frac{da}{dt} &= \\alpha a - \\omega b - (a^2 + b^2)\\, a + \\kappa(u)\\, a \\\\
\\frac{db}{dt} &= \\omega a + \\alpha b - (a^2 + b^2)\\, b + \\kappa(u)\\, b
\\end{aligned}
```

All operations are element-wise. κ(u) multiplies each oscillator's own state, and the model has no
pairwise cross-oscillator term. One κ value is computed for each batch column and shared by all
oscillators.

# Arguments
- `N`: Number of oscillators. The state dimension is `2N`.
- `M`: Dimension of the (encoded) input `u`.
- `hidden_dim`: Hidden width of the κ network (default 16).
- `init_alpha`, `init_omega`: Initialisers for the per-oscillator bifurcation parameters α (`N×1`)
  and natural frequencies ω (`N×1`).

# Parameters
- `α`: Bifurcation parameters (`N×1`).
- `ω`: Natural frequencies (`N×1`).
- `kappa`: Parameters of the coupling network `κ: ℝᴹ → ℝ`,
  `Chain(Dense(M => hidden_dim, tanh), Dense(hidden_dim => 1))`.

If `u === nothing`, κ = 0 and the drift reduces to `N` uncoupled Hopf oscillators.

# Example
```julia
drift = CoupledOscillators(8, n_control)                  # 8 oscillators → 16 latent states
diffusion = Scale(16, sigmoid)
dynamics = SDE(drift, drift_aug, diffusion, EulerHeun(); saveat=ts, dt=ts[2]-ts[1])
```
"""
struct CoupledOscillators{K, F1, F2} <: AbstractExplicitContainerLayer{(:kappa,)}
    N::Int          # Number of oscillators
    M::Int          # Input dimension
    kappa::K        # Input-dependent scalar coupling κ(u)
    init_alpha::F1  # Initialiser for the bifurcation parameters α
    init_omega::F2  # Initialiser for the natural frequencies ω
end

function CoupledOscillators(N::Int, M::Int; hidden_dim::Int=16, init_alpha=Lux.rand32, init_omega=Lux.rand32)
    kappa = Chain(Dense(M => hidden_dim, tanh), Dense(hidden_dim => 1))
    return CoupledOscillators(N, M, kappa, init_alpha, init_omega)
end

function Lux.initialparameters(rng::AbstractRNG, l::CoupledOscillators)
    α = l.init_alpha(rng, l.N, 1)
    ω = l.init_omega(rng, l.N, 1)
    kappa = Lux.initialparameters(rng, l.kappa)
    return (α=α, ω=ω, kappa=kappa)
end

# Lux unwraps the states of single-layer containers; keep them under `kappa` to mirror the parameters.
Lux.initialstates(rng::AbstractRNG, l::CoupledOscillators) = (kappa=Lux.initialstates(rng, l.kappa),)

Lux.parameterlength(l::CoupledOscillators) = 2 * l.N + Lux.parameterlength(l.kappa)
Lux.statelength(l::CoupledOscillators) = Lux.statelength(l.kappa)

# The Hopf normal form shared by both methods, plus an optional coupling term κ .* (a, b).
function _hopf_drift(a, b, α, ω)
    r² = a .^ 2 .+ b .^ 2
    da = α .* a .- ω .* b .- r² .* a
    db = ω .* a .+ α .* b .- r² .* b
    return da, db
end

function (l::CoupledOscillators)(x, u::AbstractArray, t, p, st)
    N = l.N
    a = x[1:N, :]
    b = x[N+1:2N, :]
    κ, st_kappa = l.kappa(u, p.kappa, st.kappa)   # 1 × batch
    da, db = _hopf_drift(a, b, p.α, p.ω)
    return vcat(da .+ κ .* a, db .+ κ .* b), merge(st, (kappa=st_kappa,))
end

function (l::CoupledOscillators)(x, u::Nothing, t, p, st)
    N = l.N
    a = x[1:N, :]
    b = x[N+1:2N, :]
    da, db = _hopf_drift(a, b, p.α, p.ω)
    return vcat(da, db), st
end

# CoupledOscillators is a Lux container layer, not a `DynamicalSystem`, so it needs its own
# `dxdt_u` methods to get (x, u) separately instead of vcat(x, u).
dxdt_u(model::CoupledOscillators, x, u, t, p, st) = model(x, u, t, p, st)
dxdt_u(model::CoupledOscillators, x, u::Nothing, t, p, st) = model(x, nothing, t, p, st)
