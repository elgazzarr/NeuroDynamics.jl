function create_model(dynamics::String, sde::Bool, ts, dims, config, dev)
    rng = Random.MersenneTwister(2)
    obs_encoder = Recurrent_Encoder(dims.n_neurons, config.N, config.C, config.o_encoder.hidden, config.t₀)
    init_map = Dense(config.N, config.N)
    if dynamics == "WilsonCowan"
        drift =  ModernWilsonCowan(config.N, dims.n_stimuli)
    elseif dynamics == "SlOscillators"
        drift = SlOscillators(config.N, dims.n_stimuli)
    elseif dynamics == "Neural"
        drift = Dense(config.N + dims.n_stimuli => config.N, tanh)
    elseif dynamics == "Neural_deep"
        drift = Chain(Dense(config.N + dims.n_stimuli => 64, softplus),
                        Dense(64 => config.N, tanh))
    elseif dynamics == "Linear"
        drift = Dense(config.N + dims.n_stimuli => config.N)
    else err
        error("Unknown dynamics: $dataset_name 
        \n Currnet supported dynamics: Linear, WilsonCowan, SlOscillators, Neural, Neural_deep")
    end

    if sde
        drift_aug = Chain(Dense(config.N + config.C + dims.n_stimuli, 64, tanh), Dense(64, config.N, tanh))
        diffusion = Scale(config.N, sigmoid, init_weight=identity_init(gain=0.1f0))
        dynamics = SDE(drift, drift_aug, diffusion, EM(), saveat=ts, dt=(ts[2]-ts[1]))
    else 
        dynamics = ODE(drift, Euler(), saveat=ts, dt=(ts[2]-ts[1])*2)
    end
    obs_decoder = MLP_Decoder(config.N, dims.n_neurons, config.o_decoder.hidden, config.o_decoder.depth, "Poisson")   
    ctrl_encoder, ctrl_decoder = NoOpLayer(), NoOpLayer()
    model = LatentUDE(obs_encoder, ctrl_encoder, init_map, dynamics, obs_decoder, ctrl_decoder)
    p, st = Lux.setup(rng, model) 
    p = p |> ComponentArray{Float32} |> dev
    st = st |> dev 
    return model, p, st 
end
