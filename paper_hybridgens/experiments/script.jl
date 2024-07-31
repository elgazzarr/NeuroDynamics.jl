using Lux, LuxCUDA, Random, DifferentialEquations, SciMLSensitivity, ComponentArrays,
 Plots, MLUtils, OptimizationOptimisers, LinearAlgebra, Statistics, Printf, PyCall, Distributions, BenchmarkTools, Zygote
using IterTools: ncycle
using NeuroDynamics
using WandbMacros
using ArgParse 


np = pyimport("numpy");

include("../data/dataset.jl");
include("../data/utils.jl");
include("../trainer.jl");
include("../data/dataloaders.jl")
include("../setup_model.jl")




function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--dataset", "-d"
            help = "dataset name"
            required = true
        "--dynamics", "-x"
            help = "dynamics function to run"
            default = "Neural"
        "--stochastic"
            help = "stochastic dynamics (i.e. use an SDE)"
            action = :store_true
        "--latent_dim", "-n"
            help = "state dimension of the dynamcis"
            default = 10
        "--gpu" 
            help = "Use gpu during training"
            action = :store_true
    end
    return parse_args(s)
end




function main()
    args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in args
        println("  $arg  =>  $val")
    end
    println(args["gpu"])
end

main()