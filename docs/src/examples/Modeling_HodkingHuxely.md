# Modeling Hodking-Huxely with latent neural ODEs  

In this example will show how to use the latentUDE framework to model a Hodking-Huxely neuron with dynamic synaptic inputs. 


```julia
using Pkg, Revise, Lux, Random, DifferentialEquations, SciMLSensitivity, ComponentArrays, Plots, MLUtils, OptimizationOptimisers, LinearAlgebra, Statistics, Printf
using IterTools: ncycle
using NeuroDynamics
```


## 1.Generating ground truth data 


### 1.1 Simulating Synaptic Inputs 

We will use the [Tsodyks-Markram model](https://www.pnas.org/doi/full/10.1073/pnas.94.2.719) to simulate the synaptic inputs to a neuron. We will generate multiple trajectories to later drive our Hodking-Huxley neuron model.


```julia
n_samples = 64
tspan = (0.0, 500.0)
ts = range(tspan[1], tspan[2], length=100)
p =  [30, 1000, 50, 0.5, 0.005]
function TMS(x, p, t)
    v, R, gsyn = x
    tau, tau_u, tau_R, v0, gmax = p 
    dx₁ = -(v / tau_u)
    dx₂ = (1 - R) / tau_R
    dx₃ = -(gsyn / tau)
    return vcat(dx₁, dx₂, dx₃)
end

function epsp!(integrator)
    integrator.u[1] += integrator.p[4] * (1 - integrator.u[1])
    integrator.u[3] += integrator.p[5] * integrator.u[1] * integrator.u[2]
    integrator.u[2] -= integrator.u[1] * integrator.u[2]
end
prob = ODEProblem(TMS, [0.0, 1.0, 0.0], tspan, p)
function prob_func(prob, i, repeat)
    t_start = rand(50:100)
    t_int = rand(50:100)
    t_end = rand(400:450)
    epsp_ts = PresetTimeCallback(t_start:t_int:t_end, epsp!, save_positions=(false, false))
    remake(prob, callback=epsp_ts)
end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
U = solve(ensemble_prob, Tsit5(),  EnsembleThreads(); saveat=ts, trajectories=n_samples);
plot(U, vars=(1), alpha=0.5, color=:blue, lw=0.5, legend=false, xlabel="Time (ms)", ylabel="Membrane Potential (mV)")

```

### 1.2 Simulating a Hodgkin-Huxley Neuron 


```julia
# Potassium ion-channel rate functions
alpha_n(v) = (0.02 * (v - 25.0)) / (1.0 - exp((-1.0 * (v - 25.0)) / 9.0))
beta_n(v) = (-0.002 * (v - 25.0)) / (1.0 - exp((v - 25.0) / 9.0))

# Sodium ion-channel rate functions
alpha_m(v) = (0.182 * (v + 35.0)) / (1.0 - exp((-1.0 * (v + 35.0)) / 9.0))
beta_m(v) = (-0.124 * (v + 35.0)) / (1.0 - exp((v + 35.0) / 9.0))

alpha_h(v) = 0.25 * exp((-1.0 * (v + 90.0)) / 12.0)
beta_h(v) = (0.25 * exp((v + 62.0) / 6.0)) / exp((v + 90.0) / 12.0)



function HH(x, p, t, u)
    gK, gNa, gL, EK, ENa, EL, C, ESyn, i = p
    v, n, m, h = x
    ISyn(t) = u[i](t)[end] * (ESyn - v)

    dx₁ = ((gK * (n^4.0) * (EK - v)) + (gNa * (m^3.0) * h * (ENa - v)) + (gL * (EL - v)) + ISyn(t)) / C
    dx₂ = (alpha_n(v) * (1.0 - n)) - (beta_n(v) * n)
    dx₃ = (alpha_m(v) * (1.0 - m)) - (beta_m(v) * m)
    dx₄ = (alpha_h(v) * (1.0 - h)) - (beta_h(v) * h)

    dx = vcat(dx₁, dx₂, dx₃, dx₄)
end

dxdt(x, p, t) = HH(x, p, t, U)

p = [35.0, 40.0, 0.3, -77.0, 55.0, -65.0, 1, 0, 1] 
# n, m & h steady-states
n_inf(v) = alpha_n(v) / (alpha_n(v) + beta_n(v))
m_inf(v) = alpha_m(v) / (alpha_m(v) + beta_m(v))
h_inf(v) = alpha_h(v) / (alpha_h(v) + beta_h(v))

v0 = -60
x0 = [v0, n_inf(v0), m_inf(v0), h_inf(v0)]
prob = ODEProblem(dxdt, x0, tspan, p)
prob_func(prob, i, repeat) = remake(prob, p=(p[1:end-1]..., i))
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
Y = solve(ensemble_prob, EnsembleThreads(); saveat=ts, trajectories=n_samples)
plot(Y, vars=1, label="v")

```

### 1.3 Creating a dataset and splitting it into train val test sets


```julia
Y_data = Array(Y) .|> Float32
U_data = Array(U) .|> Float32
input_dim = size(U_data)[1]
obs_dim = size(Y_data)[1]
(u_train, y_train), (u_val, y_val) = splitobs((U_data, Y_data); at=0.8, shuffle=true)
# Create dataloaders
train_loader = DataLoader((U_data, Y_data), batchsize=32, shuffle=false)
val_loader = DataLoader((U_data, Y_data), batchsize=32, shuffle=true);

```

## 2. Creating the model 


```julia
function create_model(n_states, ctrl_dim, obs_dim, context_dim, t_init)
    rng = Random.MersenneTwister(1234)
    obs_encoder = Recurrent_Encoder(obs_dim, n_states, context_dim, 32, t_init)
    vector_field = Chain(Dense(n_states+ctrl_dim, 32, softplus), Dense(32, n_states, tanh))
    dynamics = ODE(vector_field, Euler(); saveat=ts, dt=2.0)
    obs_decoder = Linear_Decoder(n_states, obs_dim, "None")   

    model = LatentUDE(obs_encoder=obs_encoder, dynamics=dynamics, obs_decoder=obs_decoder)
    p, st = Lux.setup(rng, model)
    p = p |> ComponentArray{Float32}
    return model, p, st
end
```


```julia
latent_dim = 8
context_dim = 0 # No need for context if we have ODE dynamics
t_init = 50
model, p, st = create_model(latent_dim, input_dim, obs_dim, context_dim, t_init)
u, y = first(train_loader)
ts = ts |> Array{Float32};
```

## 3. Train the model via variational inference


```julia
function train(model, p, st, train_loader, val_loader, epochs, print_every)
    
    epoch = 0
    L = frange_cycle_linear(epochs+1, 0.0f0, 1.0f0, 1, 0.5)
    losses = []
    best_model_params = nothing
    best_metric = Inf
    function loss(p, u, y, ts=ts)
        ŷ, û, x̂₀, _ = model(y, u, ts, p, st)
        batch_size = size(y)[end]
        recon_loss = mse(ŷ[1:1, :, :], y[1:1, :, :])/batch_size
        kl_loss = kl_normal(x̂₀[1], x̂₀[2])/batch_size
        l =  0.1*recon_loss + L[epoch+1]*kl_loss
        return l, recon_loss, kl_loss
    end


    callback = function(opt_state, l, recon_loss, kl_loss)
        θ = opt_state.u
        push!(losses, l)
        if length(losses) % length(train_loader) == 0
            epoch += 1
        end

        if length(losses) % (length(train_loader)*print_every) == 0
            @printf("Current epoch: %d, Loss: %.2f, Reconstruction: %d, KL: %.2f\n", epoch, losses[end], recon_loss, kl_loss)
            u, y = first(val_loader)
            batch_size = size(y)[end]
            ŷ, _, x = predict(model, y, u, ts, θ, st, 20)
            ŷ_mean = dropdims(mean(ŷ, dims=4), dims=4)
            val_mse = mse(ŷ_mean[1:1, :, :], y[1:1, :, :])
            @printf("Validation MSE: %.2f\n", val_mse)
            if val_mse < best_metric
                best_metric = val_mse
                @printf("Saving model with best metric: %.2f\n", best_metric)
                best_model_params = copy(θ)

            end

            pl = plot(transpose(y[1:1, :, 1]), label="True", lw=2.0)
            plot!(pl, transpose(ŷ_mean[1:1, :, 1]), label="Predicted", lw=2.0, xlabel="Time (ms)", ylabel="Membrane Potential (mV)")
            display(pl)
        
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ , u, y) -> loss(p, u, y), adtype)
    optproblem = OptimizationProblem(optf, p)
    result = Optimization.solve(optproblem, ADAMW(1e-3), ncycle(train_loader, epochs); callback)
    return result, losses, model, best_model_params
    
end

```


```julia
result, losses, model, best_p = train(model, p, st, train_loader, val_loader, 5000, 50)

```
