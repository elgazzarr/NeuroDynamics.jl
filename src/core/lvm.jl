"""
    LatentUDE(obs_encoder, ctrl_encoder, dynamics, obs_decoder, ctrl_decoder)

Constructs a Latent Universal Differential Equation model.

Arguments:

  - `obs_encoder`: A function that encodes the observations `y` to get the initial hidden state `x₀` and context for the dynamics if needed (Partial observability) 
  - `ctrl_encoder`: A function that encodes (high-dimensional) inputs/controls to a lower-dimensional representation if needed.
  - `init_map`: A function that maps the sampled initial conditions to plausible valubles for the dynamics. 
  - `dynamics`: A function that models the dynamics of the system (your ODE/SDE).
  - `obs_decoder`: A function that decodes the hidden states `x` to the observations `y`.
  - 'ctrl_decoder': A function that decodes the control representation to the original control space if needed.

"""
@with_kw struct LatentUDE <: LatentVariableModel
    obs_encoder = Identity_Encoder()
    ctrl_encoder = NoOpLayer()
    init_map = NoOpLayer()
    dynamics
    obs_decoder = Identity_Decoder()
    ctrl_decoder = NoOpLayer()
end 

"""
    (model::LatentUDE)(y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple)

The forward pass of the LatentUDE model.

Arguments:

  - `y`: Observations
  - `u`: Control inputs
  - `ts`: Time points
  - `ps`: Parameters
  - `st`: NamedTuple of states 

Returns:

  - `ŷ`: Decoded observations from the hidden states.
  - `ū`: Decoded control inputs from the hidden states.
  - `x̂₀`: Encoded initial hidden state.
  - `kl_path`: KL divergence path. (Only for SDE dynamics, otherwise `nothing`)ƒ
"""
function (model::LatentUDE)(y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple)
       forward!(model, y, u, ts, ps, st, model.dynamics)
end


""" 
    forward!(model::LatentUDE, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, dynamics::SDE)

The forward pass of the LatentSDE model.
"""
function forward!(model::LatentUDE, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, dynamics::SDE)
    x̂₀, context = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1]
    x₀ = model.init_map(sample_rp(x̂₀), ps.init_map, st.init_map)[1]
    x₀_aug = CRC.@ignore_derivatives fill!(similar(x₀, 1, size(x₀)[2]), 0.0f0)
    x₀  = vcat(x₀, x₀_aug)
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x_sol = dynamics(x₀, u_enc, context, ts, ps.dynamics, st.dynamics)[1]
    x_arr = cat(x_sol.u..., dims = 3)
    x_ = permutedims(x_arr, (1, 3, 2))
    x = x_[1:end-1, :, :]
    kl_path = x_[end, :, :]
    ŷ = model.obs_decoder(x, ps.obs_decoder, st.obs_decoder)[1]
    û = model.ctrl_decoder(x, ps.ctrl_decoder, st.ctrl_decoder)[1]
    return ŷ, û, x̂₀, kl_path 
end



""" 
    forward!(model::LatentUDE, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, dynamics::ODE)

The forward pass of the LatentODE model.
"""
function forward!(model::LatentUDE, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, dynamics::ODE)
    x̂₀, _ = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1] 
    x₀ = model.init_map(sample_rp(x̂₀), ps.init_map, st.init_map)[1]
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x_sol = dynamics(x₀, u_enc, ts, ps.dynamics, st.dynamics)[1]
    x_arr = cat(x_sol.u..., dims = 3)
    x = permutedims(x_arr, (1, 3, 2))
    kl_path = nothing
    ŷ = model.obs_decoder(x, ps.obs_decoder, st.obs_decoder)[1]
    û = model.ctrl_decoder(x, ps.ctrl_decoder, st.ctrl_decoder)[1]
    return ŷ, û, x̂₀, kl_path 
end




"""
    predict(model::LatentUDE, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int)

Samples trajectories from the LatentUDE model.

Arguments:

  - `model`: The `LatentUDE` model to sample from.
  - `y`: Observations used to encode the initial hidden state. 
  - `u`: Inputs for the input encoder. Can be `Nothing` or an array.
  - `ts`: Array of time points at which to sample the trajectories.
  - `ps`: Parameters for the model.
  - `st`: NamedTuple of states for different components of the model.
  - `n_samples`: Number of samples used to make the prediction.

Returns:

  - `ŷ`: Decoded observations from the sampled hidden states * `n_samples`.
  - `ū`: Decoded control inputs from the sampled hidden states * `n_samples`.
  - `x`: Sampled hidden state trajectories * `n_samples`.
"""
function predict(model::LatentUDE, y::AbstractArray, u::Union{Nothing, AbstractArray}, ts::AbstractArray, ps::ComponentArray, st::NamedTuple, n_samples::Int, dev)
    x̂₀, _ = model.obs_encoder(y, ps.obs_encoder, st.obs_encoder)[1] 
    u_enc = model.ctrl_encoder(u, ps.ctrl_encoder, st.ctrl_encoder)[1]
    x = sample_dynamics(model.dynamics, model.init_map, x̂₀, u_enc, ts, ps, st, n_samples, dev)
    ŷ = model.obs_decoder(x, ps.obs_decoder, st.obs_decoder)[1]
    û = model.ctrl_decoder(x, ps.ctrl_decoder, st.ctrl_decoder)[1]
    return ŷ, û, x
end 

