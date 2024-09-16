
function SlOscillators(N::Int, M::Int)
    @compact(σ=glorot_normal(N),
               ω=rand32(N),
               K=glorot_normal(N, N),
               B=glorot_normal(N, M),
               name="SlOscillators (N=$N, M=$M)") do xu
       
       z, u = xu
       x_ = @view z[1:N,:]
       y_ = @view z[N+1:2N, :]

       dx = (-ω.*y_ + x_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2) + K * x_ + B * sin.(u))
       dy = (ω.*x_ + y_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2)  + K * y_ + B * cos.(u))
       @return vcat(dx, dy)
   end
end
##########################                             
function WilsonCowan(N_E::Int, N_I::Int, M::Int)
    @compact(λ_E=rand32(N_E),  # Parameters for excitatory populations
             θ_E=rand32(N_E),  # Sigmoid threshold for excitatory populations
             λ_I=rand32(N_I),  # Parameters for inhibitory populations
             θ_I=rand32(N_I),  # Sigmoid threshold for inhibitory populations
             raw_w_EE=rand32(N_E, N_E),  # Raw weights from E to E
             raw_w_EI=rand32(N_E, N_I),  # Raw weights from I to E
             raw_w_IE=randn32(N_I, N_E),  # Raw weights from E to I
             raw_w_II=rand32(N_I, N_I),  # Raw weights from I to I
             w_EU=rand32(N_E, M),  # External input weights for excitatory populations
             w_IU=rand32(N_I, M),  # External input weights for inhibitory populations
             name="WilsonCowan (N = $N_E Ex + $N_I In)") do xu
                
        # State variables
        x, u = xu
        x_ex = @view x[1:N_E, :]
        x_in = @view x[N_E+1:N_E+N_I, :]

        # Sigmoid function
        S(x, λ, θ) = 1.f0 ./ (1.f0 .+ exp.(-λ .* (x .- θ)))

        # Normalize weights using softmax along the correct dimension
        w_EE = softmax(raw_w_EE, dims=2)
        w_EI = softmax(raw_w_EI, dims=2)
        w_IE = softmax(raw_w_IE, dims=2)
        w_II = softmax(raw_w_II, dims=2)

        # Differential equations for excitatory populations
        input_ex = w_EE * x_ex .- w_EI * x_in .+ w_EU * u
        dx_ex = -x_ex .+ S(input_ex, λ_E, θ_E)

        # Differential equations for inhibitory populations
        input_in = w_IE * x_ex .- w_II * x_in .+ w_IU * u
        dx_in = -x_in .+ S(input_in, λ_I, θ_I)

        # Return the concatenated derivatives
        @return vcat(dx_ex, dx_in)
    end
end 



###########################################

function JensenRit(N_E::Int, N_I::Int, N_P::Int, M::Int)
    @compact(a=[6.f0],  # Inverse time constant for excitatory neurons
             b=[20.f0],  # Inverse time constant for inhibitory neurons
             A=[3.25f0],  # Amplitude of excitatory PSP
             B=[22.0f0],  # Amplitude of inhibitory PSP
             C1=rand32(N_P),  # Connectivity parameter for pyramidal neurons
             C2=rand32(N_E),  # Connectivity parameter for excitatory neurons
             C3=rand32(N_P),  # Connectivity parameter for pyramidal neurons
             C4=rand32(N_I),  # Connectivity parameter for inhibitory neurons
             P=rand32(N_P, M),  # External input for pyramidal neurons
             name="JensenRit (N_P = $N_P Pyr, N_E = $N_E Ex, N_I = $N_I In)") do xu

        x, u = xu
        # Ensure the state vector dimension matches the sum of populations
        @assert size(x, 1) == 2 * (N_P + N_E + N_I)

        # State variables
        y0 = @view x[1:N_P, :]  # Pyramidal neurons
        y1 = @view x[N_P+1:N_P+N_E, :]  # Excitatory interneurons
        y2 = @view x[N_P+N_E+1:N_P+N_E+N_I, :]  # Inhibitory interneurons

        # First-order derivatives (velocities)
        dy0 = @view x[N_P+N_E+N_I+1:2*N_P+N_E+N_I, :]
        dy1 = @view x[2*N_P+N_E+N_I+1:2*N_P+2*N_E+N_I, :]
        dy2 = @view x[2*N_P+2*N_E+N_I+1:2*N_P+2*N_E+2*N_I, :]
       #println(size(P*u))
        # Second-order derivatives (accelerations)
        d2y0 = A .* a .* sigmoid.(y1 .- y2) + P * u .- 2.0f0 .* a .* dy0 .- (a .^ 2) .* y0
        d2y1 = A .* a .* (C2 .* sigmoid.(C1 .* y0)) .- 2.0f0 .* a .* dy1 .- (a .^ 2) .* y1
        d2y2 = B .* b .* C4 .* sigmoid.(C3 .* y0) .- 2.0f0 .* b .* dy2 .- (b .^ 2) .* y2

        # Return the concatenated derivatives
        @return vcat(dy0, dy1, dy2, d2y0, d2y1, d2y2)
    end
end


################################
function NN(N::Int, M::Int)
    @compact(l = Chain(Dense(N+M, 64, swish), Dense(64, N, swish))) do xu
        out = l(vcat(xu...))
        @return out
    end
end