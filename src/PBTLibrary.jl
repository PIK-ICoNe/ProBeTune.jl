"""
This submodule contains useful implementations of the various parts of a probabilistic tuning problem.
"""
module PBTLibrary


"""
Quick callback function to be used in the sciml_train optimization process. Displays current loss.
"""
function basic_bac_callback(p, loss)
    display(loss)
    return false
end

function benchmark_callback(p, loss, tempt, templ, initial_time)
    if Base.Libc.time() - initial_time  >= 600
        return true
    else
        push!(templ, loss)
        display(loss)
        push!(tempt,Base.Libc.time())
        return false
    end
end

export confidence_interval
function confidence_interval(losses, delta)
    N = length(losses)+4
    d = (sum(losses[:].<delta)+2)/N
    c_high=d+(d*(1-d)/N)^0.5
    c_low=d-(d*(1-d)/N)^0.5
    return d, c_low, c_high
end


end