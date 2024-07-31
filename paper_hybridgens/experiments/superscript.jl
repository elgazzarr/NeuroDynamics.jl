using ArgParse, Lux, CUDA
include("../HybridGens.jl")
using .HybridGens
using WandbMacros

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
        "--gpu" 
            help = "Use gpu during training"
            action = :store_true
        "--dataset_kwargs"
            help = "kwargs for the dataset (e.g which seesion, brain region, etc.)"
            default=Dict(:session => 11, :rois => ["MOp"])
    end
    return parse_args(s)
end



function main()
    println("Main ...")
    args = parse_commandline()
    dev = args["gpu"] ? gpu_device() : cpu_device() 

    exp_config = Dict(Symbol(k) => v for (k, v) in args)
    dataset_name = args["dataset"]; dynamics = args["dynamics"];
    diffeq = args["stochastic"] ? "sde" : "ode"
    @wandbinit project="HybridGen_Neuro" name="$dataset_name-$dynamics-$diffeq"
    @wandbconfig exp_config...

    model_config = (N = 8, C = 32, t₀=50, o_encoder=(hidden=32,), o_decoder = (hidden=64, depth=1))
    train_config = (lr=1e-3, epochs=50, log_freq=10)

    train_loader, val_loader, ts, dims = prepare_dataloaders(dataset_name, dev; batch_size=256, args["dataset_kwargs"]...);
    model, θ, st = create_model(dynamics, args["stochastic"], ts, dims, model_config, dev);
    θ_best = train(model, θ, st, train_loader, val_loader, train_config, ts, dev);
    @wandbfinish
end


main()