using Hyperopt, Wandb

function f(x, a, b; c)
  return sum(@. x + (a - 3) ^ 2 + (b ? 10 : 20) + (c - 100) ^ 2) # Function to minimize
end

# This function dispatch must be present
# `lg` can be `WandbBackend` if using that with HyperParameter Sweep
function f(lg::WandbLogger, config)
  res = f(config["x"], config["a"], config["b"]; c = config["c"])
  Wandb.log(lg, Dict("result" => res))
  return res
end

hpsweep = WandbHyperParameterSweep()

# Main macro. The first argument to the for loop is always interpreted as the number of iterations
ho = @hyperopt for i = 50, sampler = RandomSampler(), # This is default if none provided
                   a = LinRange(1,5,1000), b = [true, false],
                   c = exp10.(LinRange(-1,3,1000))
    hpsweep(f, Dict("a" => a, "b" => b, "c" => c), project = "Wandb.jl",
            config = Dict("x" => 100))
end