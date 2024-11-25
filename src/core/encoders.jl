
"""
    Encoder

An encoder is a container layer that contains three sub-layers: `linear_net`, `init_net`, and `context_net`.

# Fields

- `linear_net`: A layer that maps the input to a hidden representation.
- `init_net`: A layer that maps the hidden representation to the initial hidden state.
- `context_net`: A layer that maps the hidden representation to the context.

"""
struct Encoder  <: Lux.AbstractExplicitContainerLayer{(:linear_net, :init_net, :context_net)}
    linear_net 
    init_net
    context_net
end


"""
    (model::Encoder)(x::AbstractArray, p::ComponentVector, st::NamedTuple)

The forward pass of the encoder.

Arguments:

- `x`: The input to the encoder (e.g. observations).
- `p`: The parameters.
- `st`: The state of the encoder.

returns:

    - `x̂₀`: The initial hidden state.
    - `context`: The context.

"""
function(model::Encoder)(x, p, st)
    x, st1 = model.linear_net(x, p.linear_net, st.linear_net)
    x̂₀, st2 = model.init_net(x, p.init_net, st.init_net)
    context, st3 = model.context_net(x, p.context_net, st.context_net)
    st = (st1, st2, st3)
    return (x̂₀, context), st
end


"""
    Identity_Encoder()

Constructs an identity encoder. Useful for fully observable systems.
    
"""
function Identity_Encoder()
    linear_net = Lux.NoOpLayer()
    init_net = Lux.SelectDim(2, 1)
    context_net = Lux.NoOpLayer()
    return Encoder(linear_net, init_net, context_net)
end


"""
    Recurrent_Encoder(obs_dim, latent_dim, context_dim, hidden_dim, t_init)

Constructs a recurrent encoder.

Arguments:

- `obs_dim`: Dimension of the observations.
- `latent_dim`: Dimension of the latent space.
- `context_dim`: Dimension of the context.
- `hidden_dim`: Dimension of the hidden state.
- `t_init`: Number of initial time steps to use for the initial hidden state.

"""
function Recurrent_Encoder(obs_dim, latent_dim, context_dim, hidden_dim, t_init)
    linear_net = Dense(obs_dim => hidden_dim, tanh)
    init_net = Chain(
                    x -> reverse(x[:,1:t_init,:], dims=2),
                    Recurrence(LSTMCell(hidden_dim=>hidden_dim)),
                    BranchLayer(Dense(hidden_dim => latent_dim), Dense(hidden_dim => latent_dim, softplus)))
    
    if context_dim == 0
        context_net = Lux.NoOpLayer()
    else
        context_net = Chain(
                        x -> reverse(x, dims=2),
                        Recurrence(LSTMCell(hidden_dim=>context_dim); return_sequence=true),
                        x -> stack(x; dims=2))
    end
    
    return Encoder(linear_net, init_net, context_net)

end
