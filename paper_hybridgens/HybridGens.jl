module HybridGens
using Pkg, Revise
using Lux, LuxCUDA, Random, DifferentialEquations, SciMLSensitivity, ComponentArrays,
 Plots, MLUtils, OptimizationOptimisers, LinearAlgebra, Statistics, Printf, PyCall, Distributions, BenchmarkTools, Zygote
using IterTools: ncycle
using NeuroDynamics
using WandbMacros
np = pyimport("numpy");

include("data/dataset.jl");
include("data/utils.jl");
include("data/dataloaders.jl")

export get_dataset, prepare_dataloaders

include("trainer.jl");
export train

include("setup_model.jl")
export create_model

end