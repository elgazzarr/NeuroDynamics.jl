# Checkpointing
function save_checkpoint(state::NamedTuple;filename::String)
    @assert last(splitext(filename))==".jld2" "Filename should have a .jld2 extension."
    save(filename; state)
    return nothing
end

function _symlink_safe(src, dest)
    rm(dest; force=true)
    return symlink(src, dest)
end

function load_checkpoint(fname::String)
    try
        return JLD2[:state]
    catch
        @warn "$fname could not be loaded. This might be because the file is absent or is \
               corrupt. Proceeding by returning `nothing`."
        return nothing
    end
end