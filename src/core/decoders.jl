"""
    Decoder

A decoder is a function that takes a latent variable and produces an output (Observations or Control inputs).    

"""
struct Decoder{ON} <: Lux.AbstractExplicitContainerLayer{(:output_net,)} 
    output_net::ON
end

"""
    (model::Decoder)(x::AbstractArray, p::ComponentVector, st::NamedTuple)


The forward pass of the decoder.

Arguments:

- `x`: The input to the decoder.
- `p`: The parameters.
- `st`: The state.

returns:

    - 'ŷ': The output of the decoder.
    - 'st': The state of the decoder.

"""
function(model::Decoder)(x, p, st)
    ŷ, st = model.output_net(x, p, st)
    return ŷ, st
end


"""
    Identity_Decoder()

Constructs an identity decoder. Useful for fully observable systems.

"""
function Identity_Decoder()
    output_net = Lux.NoOpLayer()
    return Decoder(output_net)
end


"""
    Nothing_Decoder() 

Constructs a decoder that does nothing. 

"""
function Nothing_Decoder()
    output_net = WrappedFunction(x -> nothing)
    return Decoder(output_net)
end

"""
    Linear_Decoder(obs_dim, latent_dim) 

Constructs a linear decoder.

Arguments:

- `obs_dim`: Dimension of the observations.
- `latent_dim`: Dimension of the latent space. 
- `noise`: Type of observation noise. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoder.
        
"""
function Linear_Decoder(latent_dim, obs_dim, noise="Gaussian")
    if noise == "Gaussian"
        output_net = BranchLayer(Dense(latent_dim, obs_dim), Dense(latent_dim, obs_dim, softplus))
    elseif noise == "Poisson"
        output_net = Chain(Dense(latent_dim, obs_dim), x -> exp.(x))
    elseif noise == "None" 
        output_net = Dense(latent_dim, obs_dim)
    else
        error("Unknown Observation noise: $dataset_name \n Currnet supported noise: Gaussian, Poisson, None")
    end
    return Decoder(output_net)
    
end


"""
    MLP_Decoder(obs_dim, latent_dim, hidden_dim, n_hidden)

Constructs an MLP decoder.

Arguments:

- `obs_dim`: Dimension of the observations.
- `latent_dim`: Dimension of the latent space.
- `hidden_dim`: Dimension of the hidden layers.
- `n_hidden`: Number of hidden layers.
- `noise`: Type of observation noise. Default is Gaussian. Options are Gaussian, Poisson, None.

returns: 

    - The decoder.
        
"""
function MLP_Decoder(latent_dim, obs_dim, hidden_dim, n_hidden, noise="Gaussian")

    mlp = Chain([Dense(latent_dim, hidden_dim, relu) for i in 1:n_hidden]...) 
    if noise == "Gaussian"
        output_net = Chain(mlp, Dense(hidden_dim, obs_dim))
    elseif noise == "Poisson"
        output_net = Chain(mlp, Dense(hidden_dim, obs_dim), x -> exp.(x))
    elseif noise == "None"
        output_net = Chain(mlp, Dense(hidden_dim, obs_dim))
    else
        error("Unknown Observation noise: $dataset_name \n Currnet supported noise: Gaussian, Poisson, None")
    end

    return Decoder(output_net)
end


