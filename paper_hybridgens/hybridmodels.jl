
function SlOscillators_hybrid(N::Int, M::Int; neural_network)
    @compact(σ=glorot_normal(N),
               ω=rand32(N),
               K=glorot_normal(N, N),
               NN = neural_network,
               name="Hybrid SlOscillators (N=$N, M=$M)") do xu
       
       z, u = xu
       x_ = @view z[1:N,:]
       y_ = @view z[N+1:2N, :]

       dx = (-ω.*y_ + x_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2) + K * x_ + NN(vcat(x_, u)))
       dy = (ω.*x_ + y_.*(σ .+ 2(x_.^2 + y_.^2) - (x_.^2 + y_.^2).^2)  + K * y_ + NN(vcat(y_, u)))
       @return vcat(dx, dy)
   end
end
##########################                             
function WilsonCowan_hybrid(N_E::Int, N_I::Int, M::Int; neural_network)
    @compact(λ_E=rand32(N_E),  # Parameters for excitatory populations
             θ_E=rand32(N_E),  # Sigmoid threshold for excitatory populations
             λ_I=rand32(N_I),  # Parameters for inhibitory populations
             θ_I=rand32(N_I),  # Sigmoid threshold for inhibitory populations
             raw_w_EE=rand32(N_E, N_E),  # Raw weights from E to E
             raw_w_EI=rand32(N_E, N_I),  # Raw weights from I to E
             raw_w_IE=randn32(N_I, N_E),  # Raw weights from E to I
             raw_w_II=rand32(N_I, N_I),  # Raw weights from I to I
             NN = neural_network,
             name="Hybrid WilsonCowan (N = $N_E Ex + $N_I In)") do xu
                
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
        
        ext_ex, ext_in = NN((vcat(x_ex, u), vcat(x_in, u)))
        # Differential equations for excitatory populations
        input_ex = w_EE * x_ex .- w_EI * x_in .+ ext_ex
        dx_ex = -x_ex .+ S(input_ex, λ_E, θ_E)

        # Differential equations for inhibitory populations
        input_in = w_IE * x_ex .- w_II * x_in .+ ext_in
        dx_in = -x_in .+ S(input_in, λ_I, θ_I)

        # Return the concatenated derivatives
        @return vcat(dx_ex, dx_in)
    end
end 



###########################################

function JensenRit_hybrid(N_E::Int, N_I::Int, N_P::Int, M::Int; neural_network)
    @compact(a=[6.f0],  # Inverse time constant for excitatory neurons
             b=[20.f0],  # Inverse time constant for inhibitory neurons
             A=[3.25f0],  # Amplitude of excitatory PSP
             B=[22.0f0],  # Amplitude of inhibitory PSP
             C1=rand32(N_P),  # Connectivity parameter for pyramidal neurons
             C2=rand32(N_E),  # Connectivity parameter for excitatory neurons
             C3=rand32(N_P),  # Connectivity parameter for pyramidal neurons
             C4=rand32(N_I),  # Connectivity parameter for inhibitory neurons
             NN = neural_network,
             name="Hybrid JensenRit (N_P = $N_P Pyr, N_E = $N_E Ex, N_I = $N_I In)") do xu

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
        # Second-order derivatives (accelerations)
        d2y0 = A .* a .* sigmoid.(y1 .- y2) + NN(vcat(y0, u)) .- 2.0f0 .* a .* dy0 .- (a .^ 2) .* y0
        d2y1 = A .* a .* (C2 .* sigmoid.(C1 .* y0)) .- 2.0f0 .* a .* dy1 .- (a .^ 2) .* y1
        d2y2 = B .* b .* C4 .* sigmoid.(C3 .* y0) .- 2.0f0 .* b .* dy2 .- (b .^ 2) .* y2

        # Return the concatenated derivatives
        @return vcat(dy0, dy1, dy2, d2y0, d2y1, d2y2)
    end
end




###################################################
