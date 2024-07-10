# Creating a Latent SDE with differentiable drift and diffusion functions


```julia
using NeuroDynamics, Lux, Random, Plots, DifferentialEquations, ComponentArrays
```

Below we will create an example forced latent SDE with differentiable drift and diffusion functions.

For the encoder, we will use a `Recurrent_Encoder` which will take the input sequence and return the hidden state of the RNN at the last time step. This hidden state will be used as the initial condition for the SDE solver. It will also return a context vector which will be used to condition augmented SDE. 

The generative SDE will be defined with a `ModernWilsonCowan` drift and a 1 layer network for the diffusion.
The augmented SDE will have an MLP for the drift and share the same diffusion with the generative SDE. 

The decoder is an MLP with `Poisson` noise. 



```julia
obs_dim = 100
ctrl_dim = 10
dev = cpu_device()

#Hyperparameters
hp = Dict("n_states" => 10, "hidden_dim" => 64, "context_dim" => 32, "t_init" => 50)

#Encoder
obs_encoder = Recurrent_Encoder(obs_dim, hp["n_states"], hp["context_dim"],  hp["hidden_dim"], hp["t_init"])

#Dynamics
drift =  ModernWilsonCowan(hp["n_states"], ctrl_dim)
drift_aug = Chain(Dense(hp["n_states"] + hp["context_dim"], hp["hidden_dim"], softplus), Dense(hp["hidden_dim"], hp["n_states"], tanh))
diffusion = Dense(hp["n_states"], hp["n_states"],  sigmoid)
dynamics =  SDE(drift, drift_aug, diffusion, EulerHeun(), dt=0.1)

#Decoder
obs_decoder = MLP_Decoder(hp["n_states"], obs_dim,  hp["hidden_dim"], 1, "Poisson")   

#Model
model = LatentUDE(obs_encoder=obs_encoder, dynamics=dynamics, obs_decoder=obs_decoder, device=dev)
```


    LatentUDE(
        obs_encoder = Encoder(
            linear_net = Dense(100 => 64),  [90m# 6_464 parameters[39m
            init_net = Chain(
                layer_1 = WrappedFunction{:direct_call}(NeuroDynamics.var"#34#37"{Int64}(50)),
                layer_2 = Recurrence(
                    cell = LSTMCell(64 => 64),  [90m# 33_024 parameters[39m[90m, plus 1[39m
                ),
                layer_3 = BranchLayer(
                    layer_1 = Dense(64 => 10),  [90m# 650 parameters[39m
                    layer_2 = Dense(64 => 10, softplus),  [90m# 650 parameters[39m
                ),
            ),
            context_net = Chain(
                layer_1 = WrappedFunction{:direct_call}(NeuroDynamics.var"#35#38"()),
                layer_2 = Recurrence(
                    cell = LSTMCell(64 => 32),  [90m# 12_416 parameters[39m[90m, plus 1[39m
                ),
                layer_3 = WrappedFunction{:direct_call}(NeuroDynamics.var"#36#39"()),
            ),
        ),
        ctrl_encoder = NoOpLayer(),
        dynamics = SDE(
            drift = ModernWilsonCowan(10, 10, WeightInitializers.ones32, WeightInitializers.glorot_uniform, WeightInitializers.glorot_uniform, WeightInitializers.ones32),  [90m# 220 parameters[39m
            drift_aug = Chain(
                layer_1 = Dense(42 => 64, softplus),  [90m# 2_752 parameters[39m
                layer_2 = Dense(64 => 10, tanh_fast),  [90m# 650 parameters[39m
            ),
            diffusion = Dense(10 => 10, sigmoid_fast),  [90m# 110 parameters[39m
        ),
        obs_decoder = Decoder(
            output_net = Chain(
                layer_1 = Dense(10 => 64, relu),  [90m# 704 parameters[39m
                layer_2 = Dense(64 => 100),  [90m# 6_500 parameters[39m
                layer_3 = WrappedFunction{:direct_call}(NeuroDynamics.var"#45#47"()),
            ),
        ),
        ctrl_decoder = NoOpLayer(),
    ) [90m        # Total: [39m64_140 parameters,
    [90m          #        plus [39m2 states.

