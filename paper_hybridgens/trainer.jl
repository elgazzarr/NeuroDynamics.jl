function train(model::LatentUDE, p, st, train_loader, val_loader, config, ts, dev)
    train!(model, p, st, train_loader, val_loader, config, ts, dev, model.dynamics)
end

function plot_states(x, sample_n=1; kwargs...)
    xₘ = selectdim(dropmean(x, dims=4), 3, sample_n)
    xₛ = selectdim(dropmean(std(x, dims=4), dims=4), 3, sample_n)
    Plots.plot(transpose(xₘ), ribbon=transpose(xₛ), grid=false, alpha=0.3; kwargs...)
end

function train!(model::LatentUDE, p, st, train_loader, val_loader, config, ts, dev, dynamics::ODE)
    epoch = 0
    L = frange_cycle_linear(config.epochs+1, 0.5f0, 1.0f0, 1, 0.5f0)
    losses = []
    θ_best = nothing
    best_metric = -Inf
    stime = time()
    @info "Training ...."

   function loss(p, u, y, b)
        u = u |> dev; y = y|>dev;
        ŷ, _, x̂₀, _ = model(y, u, ts, p, st)
        batch_size = size(y)[end]
        recon_loss = -poisson_loglikelihood(ŷ, y)/ batch_size
        kl_init = kl_normal(x̂₀[1], x̂₀[2])/batch_size
        kl_loss =  kl_init
        l =  recon_loss + L[epoch+1]*kl_loss
        return l, recon_loss, kl_loss
    end


    callback = function(opt_state, l, recon_loss , kl_loss)
        θ = opt_state.u
        push!(losses, l)
        if length(losses) % length(train_loader) == 0
            epoch += 1
        end

        if length(losses) % (length(train_loader)*config.log_freq) == 0
            t_epoch = time() - stime
            @printf("Time/epoch %.2fs \t Current epoch: %d, \t Loss: %.2f, PoissonLL: %d, KL: %.2f\n", t_epoch/config.log_freq, epoch, losses[end], recon_loss, kl_loss)
            u, y = first(val_loader) 
            u = u |> dev; y = y|>dev;
            ŷ, _, _ = predict(model, y, u, ts, θ, st, 20, dev)
            ŷₘ = dropdims(mean(ŷ, dims=4), dims=4)
            val_bps = round(bits_per_spike(ŷₘ, y),digits=3)
            @printf("Validation bits/spike: %.2f\n", val_bps)
            train_loss = round(losses[end],digits=2)
            @wandblog val_bps train_loss

            if val_bps > best_metric
                best_metric = val_bps
                @wandblog best_metric
                θ_best = copy(θ)
                @printf("Saving best model\n")
            end
            stime = time()        
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ , u, y, b) -> loss(p,u,y,b), adtype)
    optproblem = OptimizationProblem(optf, p)
    result = Optimization.solve(optproblem, ADAMW(config.lr), ncycle(train_loader, config.epochs); callback)
    return model, θ_best
end


function train!(model::LatentUDE, p, st, train_loader, val_loader, config, ts, dev, dynamics::SDE)
    epoch = 0
    L = frange_cycle_linear(config.epochs+1, 0.5f0, 1.0f0, 1, 0.5f0)
    losses = []
    θ_best = nothing
    best_metric = -Inf
    stime = time()
    @info "Training ...."

   function loss(p, u, y, b)
        u = u |> dev; y = y|>dev;
        ŷ, _, x̂₀, kl_pq = model(y, u, ts, p, st)
        batch_size = size(y)[end]
        recon_loss = -poisson_loglikelihood(ŷ, y)/ batch_size
        kl_init = kl_normal(x̂₀[1], x̂₀[2])/batch_size
        kl_path = mean(kl_pq[end,:])
        kl_loss =  kl_init + kl_path
        l =  recon_loss + L[epoch+1]*kl_loss
        return l, recon_loss, kl_loss
    end


    callback = function(opt_state, l, recon_loss , kl_loss)
        θ = opt_state.u
        push!(losses, l)
        if length(losses) % length(train_loader) == 0
            epoch += 1
        end

        if length(losses) % (length(train_loader)*config.log_freq) == 0
            t_epoch = time() - stime
            @printf("Time/epoch %.2fs \t Current epoch: %d, \t Loss: %.2f, PoissonLL: %d, KL: %.2f\n", t_epoch/config.log_freq, epoch, losses[end], recon_loss, kl_loss)
            u, y, r = first(val_loader) 
            u = u |> dev; y = y|>dev;
            ŷ, _, x = predict(model, y, u, ts, θ, st, 20, dev)
            ŷₘ = dropdims(mean(ŷ, dims=4), dims=4)
            val_bps = round(bits_per_spike(ŷₘ, y),digits=3)
            @printf("Validation bits/spike: %.2f\n", val_bps)
            train_loss = round(losses[end],digits=2)
            @wandblog val_bps train_loss
            
            if val_bps > best_metric
                best_metric = val_bps
                @wandblog best_metric
                θ_best = copy(θ)
                @printf("Saving best model\n")
            end
            d = plot_states(x)
            display(d)
            stime = time()        
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ , u, y, b) -> loss(p,u,y,b), adtype)
    optproblem = OptimizationProblem(optf, p)
    result = Optimization.solve(optproblem, ADAMW(config.lr), ncycle(train_loader, config.epochs); callback)
    return model, θ_best
end