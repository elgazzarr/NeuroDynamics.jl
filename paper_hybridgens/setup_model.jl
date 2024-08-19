function create_model(dynamics::String, hybrid::Bool, sde::Bool, ts, dims, config, dev)
    N = config.N ; C = config.C; M = dims.n_stimuli
    rng = Random.MersenneTwister(2)
    obs_encoder = Recurrent_Encoder(dims.n_neurons, config.N, config.C, config.o_encoder.hidden, config.t₀)

    # Define the dynamics
    if dynamics == "SlOscillators"
        init_map = Dense(N, N*2, sigmoid)
        drift = hybrid ? SlOscillators_hybrid(N, M; neural_network=Dense(N + M, N, tanh)) : SlOscillators(N, M)
        N_drift = N*2
    elseif dynamics == "WilsonCowan"
        init_map = Dense(N, N, sigmoid)
        N_ex = trunc(Int, 0.8*N); N_in = N - N_ex
        drift = hybrid ? WilsonCowan_hybrid(N_ex,  N_in, M; neural_network=Parallel(nothing, Dense(N_ex+M, N_ex, tanh), Dense(N_in+M, N_in, tanh))) : WilsonCowan(N_ex, N_in, M)
        N_drift = N
    elseif dynamics == "JensenRit"
        N_ = trunc(Int, N/3)
        init_map = Dense(N, N_*3*2, sigmoid)
        N_in = N_ ; N_ex = N_ ; N_pyr = N_
        drift = hybrid ? JensenRit_hybrid(N_pyr, N_ex, N_in, M; neural_network=Dense(N_pyr + M, N_pyr, tanh)) : JensenRit(N_pyr, N_ex, N_in, M)
        N_drift = N_*3*2
    elseif dynamics == "Neural"
        init_map = Dense(config.N, config.N)
        drift = NN(N, M)
        N_drift = N
    else 
        error("Unknown dynamics, choose from: [SlOscillators, WilsonCowan, JensenRit, Neural]")
    end

    if sde
        drift_aug = Chain(Dense(N_drift + C + dims.n_stimuli, 64, tanh), Dense(64, N_drift, tanh))
        diffusion = Scale(N_drift, sigmoid, init_weight=identity_init(gain=0.1f0))
        dynamics = SDE(drift, drift_aug, diffusion, EM(), saveat=ts, dt=(ts[2]-ts[1])*2)
    else 
        dynamics = ODE(drift, Euler(), saveat=ts, dt=(ts[2]-ts[1]))
    end
    obs_decoder = MLP_Decoder(N_drift, dims.n_neurons, config.o_decoder.hidden, config.o_decoder.depth, "Poisson")   
    ctrl_encoder, ctrl_decoder = NoOpLayer(), NoOpLayer()
    model = LatentUDE(obs_encoder, ctrl_encoder, init_map, dynamics, obs_decoder, ctrl_decoder)
    p, st = Lux.setup(rng, model) 
    p = p |> ComponentArray{Float32} |> dev
    st = st |> dev 
    return model, p, st 
end



struct MapDecoder{SM, D} <: Lux.AbstractExplicitContainerLayer{(:state_map, :decoder,)}
    state_map::SM
    decoder::D
end

function MapDecoder(state_map::SM, decoder::D) where {SM, D}
    return MapDecoder{SM, D}(state_map, decoder)
end


function(m::MapDecoder)(x, p, st)
    x = m.state_map(x, p.state_map, st.state_map)
    ŷ, st = m.decoder(x, p.decoder, st.decoder)
    return ŷ, st
end 


function create_model_small(dynamics::String, hybrid::Bool, sde::Bool, ts, dims, config, dev)
    N = config.N ; C = config.C; M = dims.n_stimuli
    rng = Random.MersenneTwister(2)
    obs_encoder = Recurrent_Encoder(dims.n_neurons, config.N, config.C, config.o_encoder.hidden, config.t₀)

    # Define the dynamics
    if dynamics == "SlOscillators"
        init_map = Dense(N, N*2, sigmoid)
        drift = hybrid ? SlOscillators_hybrid(N, M; neural_network=Dense(N + M, N, tanh)) : SlOscillators(N, M)
        state_map = Lux.SelectDim(1, 1:N)
        N_drift = N*2
    elseif dynamics == "WilsonCowan"
        init_map = Dense(N, N, sigmoid)
        N_ex = trunc(Int, 0.8*N); N_in = N - N_ex
        drift = hybrid ? WilsonCowan_hybrid(N_ex,  N_in, M; neural_network=Parallel(nothing, Dense(N_ex+M, N_ex, tanh), Dense(N_in+M, N_in, tanh))) : WilsonCowan(N_ex, N_in, M)
        state_map = NoOpLayer()
        N_drift = N
    elseif dynamics == "JensenRit"
        N_ =  N >2 ? trunc(Int, N/3) : 1
        init_map = Dense(N, N_*3*2, sigmoid)
        N_in = N_ ; N_ex = N_ ; N_pyr = N_
        drift = hybrid ? JensenRit_hybrid(N_pyr, N_ex, N_in, M; neural_network=Dense(N_pyr + M, N_pyr, tanh)) : JensenRit(N_pyr, N_ex, N_in, M)
        state_map = Lux.SelectDim(1, 1:N)
        N_drift = N_*3*2
    elseif dynamics == "Neural"
        init_map = Dense(config.N, config.N)
        drift = NN(N, M)
        N_drift = N
        state_map = NoOpLayer()
    else 
        error("Unknown dynamics, choose from: [SlOscillators, WilsonCowan, JensenRit, Neural]")
    end

    if sde
        drift_aug = Chain(Dense(N_drift + C + dims.n_stimuli, 64, tanh), Dense(64, N_drift, tanh))
        diffusion = Scale(N_drift, sigmoid, init_weight=identity_init(gain=0.1f0))
        dynamics = SDE(drift, drift_aug, diffusion, EM(), saveat=ts, dt=(ts[2]-ts[1])*2)
    else 
        dynamics = ODE(drift, Euler(), saveat=ts, dt=(ts[2]-ts[1]))
    end
    obs_decoder = MapDecoder(state_map, MLP_Decoder(N_drift, dims.n_neurons, config.o_decoder.hidden, config.o_decoder.depth, "Poisson"))
    ctrl_encoder, ctrl_decoder = NoOpLayer(), NoOpLayer()
    model = LatentUDE(obs_encoder, ctrl_encoder, init_map, dynamics, obs_decoder, ctrl_decoder)
    p, st = Lux.setup(rng, model) 
    p = p |> ComponentArray{Float32} |> dev
    st = st |> dev 
    return model, p, st 
end





