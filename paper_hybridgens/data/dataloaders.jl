function prepare_dataloaders(dataset_name, dev; batch_size=128, augment_time=false, kwargs...)
    @info "Preparing data loader for $dataset_name"
    U_, Y_neural, Y_behaviour, ts = get_dataset(dataset_name; kwargs...)
    n_neurons, n_timepoints, n_trials = size(Y_neural)
    n_behaviour = size(Y_behaviour, 1)
    ts_input = repeat(ts, 1, n_trials)
    ts_input = reshape(ts_input, (1, size(ts_input)...))

    if augment_time
        isnothing(U_) ? U = ts_input : U = U_
    else
        isnothing(U_) ? U = zeros32(1, n_timepoints, n_trials) : U = vcat(U_, ts_input)
    end

    n_stimuli = size(U, 1)

    println("#Neurons: $n_neurons \t #Stimulus: $n_stimuli \t #Behaviour: $n_behaviour \t #Timepoints: $n_timepoints \t #Trials: $n_trials")
   
    U, Y_neural, Y_behaviour = map(x -> x |> Array{Float32}, (U, Y_neural, Y_behaviour))
    ts = ts |> Array{Float32}
    
    split_ = dataset_name == "mc_maze" ? 0.9 : 0.8
    (u_train, y_train, b_train), (u_val, y_val, b_val) = splitobs((U, Y_neural, Y_behaviour); at=split_)    
    train_loader = DataLoader((u_train, y_train, b_train), batchsize=batch_size, shuffle=true)
    val_loader = DataLoader((u_val, y_val, b_val), batchsize=32, shuffle=true)
    dims = (n_neurons = n_neurons, n_behaviour = n_behaviour, n_stimuli = n_stimuli, n_timepoints = n_timepoints, n_trials = n_trials)
    return train_loader, val_loader, ts, dims

end