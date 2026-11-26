using NeuroDynamics
using Lux, Random, ComponentArrays, DifferentialEquations
using Test

@testset "NeuroDynamics.jl" begin

    @testset "CoupledOscillators" begin
        rng = Random.MersenneTwister(0)
        N, M, B = 3, 4, 5
        drift = CoupledOscillators(N, M)
        ps, st = Lux.setup(rng, drift)

        @testset "parameters" begin
            @test size(ps.α) == (N, 1)
            @test size(ps.ω) == (N, 1)
            # α, ω plus κ-MLP: Dense(M => 16) + Dense(16 => 1)
            @test Lux.parameterlength(drift) == 2N + (M * 16 + 16) + (16 + 1)
            @test Lux.parameterlength(drift) == length(ComponentArray(ps))
        end

        @testset "batched output shape" begin
            x = randn(Float32, 2N, B)
            u = randn(Float32, M, B)
            dx, _ = drift(x, u, 0.0f0, ps, st)
            @test size(dx) == (2N, B)
            dx0, _ = drift(x, nothing, 0.0f0, ps, st)
            @test size(dx0) == (2N, B)
        end

        @testset "κ ≡ 0 reduces to Hopf normal form" begin
            α = Float32[0.5, -1.0, 2.0]
            ω = Float32[1.0, 3.0, -0.5]
            # κ-MLP with all-zero weights and biases gives κ(u) = 0 for any u.
            ps0 = (α=reshape(α, N, 1), ω=reshape(ω, N, 1),
                   kappa=zero(ComponentArray(ps.kappa)))
            x = Float32[1 0; 2 -1; -1 0.5;    # a (real parts), 2 batch columns
                        0 1; 1 2; 3 -2]      # b (imaginary parts)
            u = randn(Float32, M, 2)

            expected = similar(x)
            for j in 1:2, i in 1:N
                a, b = x[i, j], x[N+i, j]
                r2 = a^2 + b^2
                expected[i, j]   = α[i]*a - ω[i]*b - r2*a
                expected[N+i, j] = ω[i]*a + α[i]*b - r2*b
            end

            dx_u, _ = drift(x, u, 0.0f0, ps0, st)
            dx_nothing, _ = drift(x, nothing, 0.0f0, ps0, st)
            @test dx_u ≈ expected
            @test dx_nothing ≈ expected
        end

        @testset "κ(u) multiplies each oscillator's own state" begin
            x = randn(Float32, 2N, B)
            u = randn(Float32, M, B)
            κ = drift.kappa(u, ps.kappa, st.kappa)[1]
            @test size(κ) == (1, B)
            dx, _ = drift(x, u, 0.0f0, ps, st)
            dx0, _ = drift(x, nothing, 0.0f0, ps, st)
            @test dx .- dx0 ≈ κ .* x
        end

        @testset "inside SDE" begin
            M = 2
            drift = CoupledOscillators(N, M)
            drift_aug = Chain(Dense(2N + 1 + M => 8, tanh), Dense(8 => 2N, tanh))
            diffusion = Scale(2N, sigmoid)
            ts = range(0.0f0, 1.0f0; length=11)
            dynamics = SDE(drift, drift_aug, diffusion, EulerHeun(); saveat=ts, dt=ts[2] - ts[1])
            ps, st = Lux.setup(rng, dynamics)
            ps = ComponentArray{Float32}(ps)

            x0 = vcat(randn(Float32, 2N, B), zeros(Float32, 1, B))   # + KL path state
            u = randn(Float32, M, length(ts), B)
            c = randn(Float32, 1, length(ts), B)
            sol, _ = dynamics(x0, u, c, ts, ps, st)
            X = Array(sol)
            @test size(X) == (2N + 1, B, length(ts))
            @test all(isfinite, X)

            x̂₀ = (randn(Float32, 2N, B), fill(0.1f0, 2N, B))
            samples = sample_dynamics(dynamics, x̂₀, u, ts, ps, st, 2)
            @test size(samples) == (2N, length(ts), B, 2)
            @test all(isfinite, samples)
        end
    end

    @testset "HarmonicOscillators" begin
        rng = Random.MersenneTwister(1)
        N, M, B = 3, 2, 4
        model = HarmonicOscillators(N; M=M)
        ps, st = Lux.setup(rng, model)
        x = randn(Float32, 2N, B)
        u = randn(Float32, M, B)
        xp, v = x[1:N, :], x[N+1:2N, :]

        # Regression: K * x and B * u must be matrix products (M ≠ N, so the old
        # element-wise version could not even broadcast).
        dx, _ = model(x, u, 0.0f0, ps, st)
        @test size(dx) == (2N, B)
        @test dx[1:N, :] ≈ v
        @test dx[N+1:2N, :] ≈ -ps.γ .* v .- ps.ω .^ 2 .* xp .+ ps.K * xp .+ ps.B * u

        dx0, _ = model(x, nothing, 0.0f0, ps, st)
        @test size(dx0) == (2N, B)
        @test dx0[N+1:2N, :] ≈ -ps.γ .* v .- ps.ω .^ 2 .* xp .+ ps.K * xp
    end
end
