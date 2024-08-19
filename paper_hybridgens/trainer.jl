function train(model::LatentUDE, p, st, train_loader, val_loader, config, ts, dev; init_epoch=0)
    train!(model, p, st, train_loader, val_loader, config, ts, dev, model.dynamics, init_epoch)
end

function plot_states(x, sample_n=1; kwargs...)
    xₘ = selectdim(dropmean(x, dims=4), 3, sample_n)
    xₛ = selectdim(dropmean(std(x, dims=4), dims=4), 3, sample_n)
    Plots.plot(transpose(xₘ), ribbon=transpose(xₛ), grid=false, alpha=0.3; kwargs...)
end

function evaluate(model, data_loader, ts, θ, st, dev; sample, ch=0)
    u, y, r = first(data_loader)
    ch = ch == 0 ? rand(1:size(y,1)) : ch
    θ = θ |> cpu_device() 
    ŷ, _, x = predict(model, y, u, ts, θ, st, 20, cpu_device())
    ŷₘ = dropdims(mean(ŷ, dims=4), dims=4)
    ŷₛ = dropdims(std(ŷ, dims=4), dims=4)
    dist = Poisson.(ŷₘ)
    pred_spike = rand.(dist)
    xₘ = dropdims(mean(x, dims=4), dims=4)
    p1 = plot(transpose(y[ch:ch,:,sample]), label="GT spike train", lw=2, color="red", grid=false)
    p2 = plot(transpose(pred_spike[ch:ch,:,sample]), label="Predicted spike train", lw=2, color="green", grid=false)
    #p3 = plot(transpose(r[ch:ch,:,sample]), label="GT rates", lw=2, color="red")
    #p3 = plot!(p3, transpose(ŷₘ[ch:ch,:,sample]), ribbon=transpose(ŷₛ[ch:ch,:,sample]), label="Infered rates", lw=2, color="green", xlabel="time(ms)")
    p3 = plot(transpose(ŷₘ[ch:ch,:,sample]), ribbon=transpose(ŷₛ[ch:ch,:,sample]), label="Infered rates", lw=2, color="green", xlabel="time(ms)")
    p4 = plot_states(x, sample; label=false, yticks=false, xlabel="time(ms)")
    p = plot(p1, p2, p3, p4, layout=(4,1), size=(1200, 800), legend=:topright)
    return p
end

function train!(model::LatentUDE, p, st, train_loader, val_loader, config, ts, dev, dynamics::ODE, init_epoch)
    epoch = init_epoch
    L = frange_cycle_linear(init_epoch+config.epochs+1, 0.0f0, 20.0f0, 1, 0.5f0)
    θ_best = nothing; best_metric = -Inf; counter=0; count_thresh = 15;
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
        θ = opt_state.u |> cpu; st_ = st |> cpu
        if opt_state.iter % length(train_loader) == 0
            epoch += 1
            if epoch % config.log_freq == 0
                t_epoch = time() - stime
                @printf("Time/epoch %.2fs \t Current epoch: %d, \t Loss: %.2f, PoissonLL: %d, KL: %.2f\n", t_epoch/config.log_freq, epoch, l, recon_loss, kl_loss)
                val_bps = 0.f0
                for (u, y, _) in val_loader
                    u = u; y = y;
                    Eŷ, _, x = predict(model, y, u, ts, θ, st_, 20, cpu_device())
                    val_bps += mean(mapslices(ŷ -> bits_per_spike(ŷ, y), Eŷ, dims=[1,2,3]))
                end
                val_bps /= round(length(val_loader), digits=3)
                @printf("Validation bits/spike: %.3f\n", val_bps)
                train_loss = round(l, digits=3)
                @wandblog val_bps train_loss step=epoch
                if val_bps > best_metric
                    best_metric = val_bps
                    @wandblog best_metric step=epoch
                    θ_best = copy(θ)
                    save_state = (model=model, θ=θ_best, st=st, data_loader=val_loader, epoch=epoch)
                    @printf("Saving best model\n")
                    save_object(joinpath(config.save_path, "bestmodel.jld2"), save_state)
                    counter = 0
                else 
                    counter += 1
                    if counter > count_thresh
                        @printf("Early stopping at epoch: %.f\n", epoch)
                        return true
                    end
                    if counter > 5
                        Optimisers.adjust!(opt_state.original, config.lr/(counter * 2))
                        @printf("No improvment, adjusting learning rate to: %.4f\n", config.lr/(counter * 2))
                    end
                end   
                stime = time()
            end
            if epoch % config.plot_freq == 0 
                d = evaluate(model, val_loader, ts, θ, st, dev, sample=1) 
                display(d)
                image_path = joinpath(config.save_path, "img_epoch=$epoch.pdf")
                savefig(image_path)
            end
        end 
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ , u, y, b) -> loss(p,u,y,b), adtype)
    optproblem = OptimizationProblem(optf, p)
    result = Optimization.solve(optproblem, ADAMW(config.lr), ncycle(train_loader, config.epochs); callback)
    return model, θ_best
end


function train!(model::LatentUDE, p, st, train_loader, val_loader, config, ts, dev, dynamics::SDE, init_epoch)
    epoch = 0
    L = frange_cycle_linear(init_epoch+config.epochs+1, 0.0f0, 1.0f0, 1, 0.5f0)
    θ_best = nothing; best_metric = -Inf; counter=0; count_thresh = 10;
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
        if opt_state.iter % length(train_loader) == 0
            epoch += 1
            if epoch % config.log_freq == 0
                t_epoch = time() - stime
                @printf("Time/epoch %.2fs \t Current epoch: %d, \t Loss: %.2f, PoissonLL: %d, KL: %.2f\n", t_epoch/config.log_freq, epoch, l, recon_loss, kl_loss)
                val_bps = 0.f0
                e_val_bps = 0.f0
                for (u, y, r) in val_loader
                    u = u |> dev; y = y|>dev;
                    Eŷ, _, x = predict(model, y, u, ts, θ, st, 20, dev)
                    ŷₘ = dropmean(Eŷ, dims=4)
                    val_bps += bits_per_spike(ŷₘ, y)  
                    e_val_bps += mean(mapslices(ŷ -> bits_per_spike(ŷ, y), Eŷ, dims=[1,2,3]))
                end
                val_bps /= round(length(val_loader), digits=3)
                e_val_bps /= round(length(val_loader), digits=3)
                @printf("Validation bits/spike: %.3f \t Expectation %.3f\n", val_bps, e_val_bps)
                train_loss = round(l, digits=3)
                @wandblog val_bps train_loss step=epoch
                if val_bps > best_metric
                    best_metric = val_bps
                    @wandblog best_metric step=epoch
                    θ_best = copy(θ)
                    @printf("Saving best model\n")
                    save_state = (model=model, θ=θ_best |> cpu, st=st |> cpu, data_loader=val_loader, epoch=epoch)
                    save_object(joinpath(config.save_path, "bestmodel.jld2"), save_state)
                    counter = 0
                else 
                    counter += 1
                    if counter > count_thresh
                        @printf("Early stopping at epoch: %.f\n", epoch)
                        return true
                    end
                    if counter > 3
                        Optimisers.adjust!(opt_state.original, config.lr/(counter * 2))
                        @printf("No improvment, adjusting learning rate to: %.4f\n", config.lr/(counter * 2))
                    end
                end   
                stime = time()  
            end

            if epoch % config.plot_freq == 0 
                d = evaluate(model, val_loader, ts, θ, st, dev, sample=1) 
                display(d)
                image_path = joinpath(config.save_path, "img_epoch=$epoch.pdf")
                savefig(image_path)
            end

        end        
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((p, _ , u, y, b) -> loss(p,u,y,b), adtype)
    optproblem = OptimizationProblem(optf, p)
    result = Optimization.solve(optproblem, ADAMW(config.lr), ncycle(train_loader, config.epochs); callback)
    return model, θ_best
end
