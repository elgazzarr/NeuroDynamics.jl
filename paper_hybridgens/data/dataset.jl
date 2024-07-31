
"""Dataasets: 
    - synthetic
    - mc_maze
    - stienmentz
    - area2_bump
"""

datasets_path = "/home/artiintel/ahmelg/Datasets"
function get_dataset(dataset_name::String; kwargs...)
    data = np.load("$datasets_path/$dataset_name.npy", allow_pickle=true)
    if dataset_name == "synthetic"
        return get_synthetic()
    elseif dataset_name == "mc_maze"
        return get_mcmaze(data)
    elseif dataset_name == "steinmetz"
        return get_steinmetz(data; kwargs...)
    elseif dataset_name == "area2_bump"
        return get_area2bump(data)
    else
        error("Unknown dataset: $dataset_name \n 
        Currnet supported datasets: synthetic, mc_maze, steinmetz, area2_bump")
    end
end


function get_mcmaze(data)
    U = nothing
    Y_neural = permutedims(get(data[1], "spikes") , [3, 2, 1])
    Y_behaviour = permutedims(get(data[1], "hand_vel") , [3, 2, 1])
    ts = range(0.0, 2.5, length = size(Y_neural, 2))
    return  U, Y_neural, Y_behaviour, ts 
end


function get_area2bump(data)
    U = permutedims(get(data[1], "force") , [3, 2, 1])
    Y_neural = permutedims(get(data[1], "spikes") , [3, 2, 1])
    Y_behaviour = permutedims(get(data[1], "hand_pos") , [3, 2, 1])
    ts = range(0.0, 2.5, length = size(Y_neural, 2))
    return  U, Y_neural, Y_behaviour, ts 
end


function get_steinmetz(data; session, rois)
    dataset_ind = session
    data = get(data, "alldat")[dataset_ind]
    dt = get(data, "bin_size")
    areas = unique(get(data, "brain_area"))
    println("Available areas in this session are: $areas")
    con_r, con_l, go_cue = get_stim_onset_cue(data)
    spikes = get_spikes(data, rois)
    wheel_vel, response, response_time = get_behaviour(data)
    n_timepoints = size(spikes, 2)    
    ts = range(0.0, 2.5, length = n_timepoints)
    U = vcat(reshape(con_r, (1, size(con_r)...)), reshape(con_l, (1, size(con_r)...)), reshape(go_cue, (1,size(go_cue)...)))
    Y_neural = spikes 
    Y_behaviour = wheel_vel 
    return U, Y_neural, Y_behaviour, ts 
end
