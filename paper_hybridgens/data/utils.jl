

function get_spikes(dataset, rois)
    spks = get(dataset, "spks")
    spks = permutedims(spks, [1,3,2])
    all_areas = string.(get(dataset, "brain_area"))
    indices = findall(area -> area in rois, all_areas) 
    spks = spks[indices,:,:]
    return spks
end


function get_spikes_dict(dataset, rois)
    spks = get(dataset, "spks")
    spks = permutedims(spks, [1, 3, 2])
    all_areas = get(dataset, "brain_area")
    all_areas = string.(all_areas)
    rois = string.(rois)
    area_dict = Dict{String, Array{Float64, 3}}()
        for roi in rois
        indices = findall(area -> area == roi, all_areas)
        area_dict[roi] = spks[indices, :, :]
    end
    
    return area_dict
end



function get_stim_onset_cue(dataset)
    # retunrs a 3 x ntimepoints matrix with the stimulus (Left and right) and go cue 
    dt = get(dataset, "bin_size")
    n_timepoints = size(get(dataset, "spks"), 3)
    n_trials = size(get(dataset, "spks"), 2)
    ts = LinRange(0, n_timepoints*dt, n_timepoints)
    cs_r = get(dataset, "contrast_right")
    cs_l = get(dataset, "contrast_left")
    stim_onset = get(dataset, "stim_onset")
    go_cues = get(dataset, "gocue") 
    tol = dt
    find_index = x -> findfirst(y -> abs(y - x) < tol, ts)
    ind_stim = find_index(stim_onset)
    inds_go = map(find_index, go_cues)
    cr_t = zeros(n_timepoints, n_trials)
    cl_t = zeros(n_timepoints, n_trials)
    go_t = zeros(n_timepoints, n_trials)
    for (i, (c_r, c_l, go)) in enumerate(zip(cs_r, cs_l, inds_go))
        cr_t[ind_stim:end, i] .= c_r 
        cl_t[ind_stim:end, i] .= c_l
        go_t[go:end, i] .= 1
    end

    return cr_t, cl_t, go_t
end

function get_behaviour(dataset)
    wheel_velocity = get(dataset, "wheel") 
    response = get(dataset, "response")
    response_time = get(dataset, "response_time")
    return permutedims(wheel_velocity, [1,3,2]), reshape(response, (1, length(response))), permutedims(response_time, (2, 1))
end


function get_feedback(dataset)
    feedback = get(dataset, "feedback_type")
    feedback_time = get(dataset, "feedback_time")
    return reshape(feedback, (1, length(feedback))), permutedims(feedback_time, (2, 1))
end