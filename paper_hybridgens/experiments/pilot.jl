using Pkg, Revise
using WandbMacros
using Lux, LuxCUDA, Random, DifferentialEquations, SciMLSensitivity, ComponentArrays,
 Plots, MLUtils, OptimizationOptimisers, LinearAlgebra, Statistics, Printf, PyCall, Distributions, BenchmarkTools, Zygote
using IterTools: ncycle
using NeuroDynamics
np = pyimport("numpy");

include("../data/dataset.jl");
include("../data/utils.jl");
include("../trainer.jl");
include("../data/dataloaders.jl")
include("../setup_model.jl")

device = "gpu"
dev = device == "gpu" ? gpu_device() : cpu_device() 



DATASET_NAME = "area2_bump"
DATASET_KWARGS = Dict(:session => 11, :rois => ["MOp"])
DYNAMICS = "Neural"
stochastic = false

run_config = Dict("dataset" => DATASET_NAME,
                                "dynamical system"  => DYNAMICS,
                               "stochastic" => stochastic,
                               "latent_dim" => 8,
                               "device" => device
                               )

symbol_dict = Dict(Symbol(k) => v for (k, v) in run_config)

@wandbinit project="HybridGen_Neuro" name="$DATASET_NAME-$DYNAMICS"
@wandbconfig symbol_dict...


model_config = (N = 8, C = 32, t₀=100, o_encoder=(hidden=32,), o_decoder = (hidden=64, depth=1))
train_config = (lr=1e-3, epochs=100, log_freq=10)



train_loader, val_loader, ts, dims = prepare_dataloaders(DATASET_NAME, dev; batch_size=128, DATASET_KWARGS...);
model, θ, st = create_model(DYNAMICS, stochastic, ts, dims, model_config, dev);
result, θ_best = train(model, θ, st, train_loader, val_loader, train_config, dev);
@wandbfinish 
