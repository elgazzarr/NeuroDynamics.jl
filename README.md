# NeuroDynamics.jl


[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://elgazzarr.github.io/NeuroDynamics.jl/dev/)
[![Build Status](https://github.com/elgazzarr/NeuroDynamics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/elgazzarr/NeuroDynamics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/elgazzarr/NeuroDynamics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/elgazzarr/NeuroDynamics.jl)


Scalable generative modeling of neural dynamics in Julia.

NeuroDynamics.jl is a Julia package for scalable and efficient generative modeling of neural dynamics [[1](https://arxiv.org/abs/2403.14510)].
Neural systems are modeld as a system of of stochastic differential equations with differentiable drift and diffusion functions.  
The package provides a high-level interface for specifying and fitting these models to neural data using variational inference [[2](https://arxiv.org/abs/2001.01328), [3](https://arxiv.org/abs/1905.09883)] and gradient-based optimization.

## Version 1.0.0-DEV

The pacakage is still in DEV mode but if you can't wait, feel free to use it and report any issues you encounter.

## Installation

The dev version can be installed by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.dev("https://github.com/elgazzarr/NeuroDynamics.jl")
```

## Tutorials, Examples, and Documentation

For more information, check out the [documentation](https://elgazzarr.github.io/NeuroDynamics.jl/dev/).


## Example of a simple Latent SDE model

```julia
using NeuroDynamics, Lux, LuxCUDA

obs_dim = 100
ctrl_dim = 10
dev = gpu_device()

hp = Dict("n_states" => 10, "hidden_dim" => 64, "context_dim" => 32, "t_init" => 50)

obs_encoder = Recurrent_Encoder(obs_dim, hp["n_states"], hp["context_dim"],  hp["hidden_dim"], hp["t_init"])
drift =  ModernWilsonCowan(hp["n_states"], ctrl_dim)
drift_aug = Chain(Dense(hp["n_states"] + hp["context_dim"], hp["hidden_dim"], softplus), Dense(hp["hidden_dim"], hp["n_states"], tanh))
diffusion = Dense(hp["n_states"], hp["n_states"], sigmoid)
dynamics =  SDE(drift, drift_aug, diffusion, EulerHeun(), dt=0.1)
obs_decoder = MLP_Decoder(hp["n_states"], obs_dim,  hp["hidden_dim"], 1, "Poisson")   
ctrl_encoder, ctrl_decoder = NoOpLayer(), NoOpLayer()

model = LatentUDE(obs_encoder, ctrl_encoder, dynamics, obs_decoder, ctrl_decoder, dev)

```


## Citation 

If you use this package in your research, please cite the following paper:

```
@article{elgazzar2024universal,
  title={Universal Differential Equations as a Common Modeling Language for Neuroscience},
  author={ElGazzar, Ahmed and van Gerven, Marcel},
  journal={arXiv preprint arXiv:2403.14510},
  year={2024}
}
```

