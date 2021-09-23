using Plots
 
#export plot_callback
"""
Plot trajectories of the first nodes of sys and spec, and display the loss.
Parameters:
- pbt: PBT problem
- p: array of sys and spec parameters, usually res.minimizer
- loss: value of the output distance
Optional parameters:
- loss_array: array to append value of loss to. Can be used to plot progress of the optimization over iterations.
- scenario_nums: number(s) of scenarios to be plotted. If not provided, a scenario is chosen randomly.
- fig_name: path and file name to save the plot. If not provided, plot is not saved.
- offset: defines by how much each scenario is offset from the previous one for better plot readability. If set to -1, all samples are plotted on subplots instead of a single figure.
- plot_options: various plot options
"""
function plot_callback(pbt, p, loss; loss_array=nothing, scenario_nums=nothing, fig_name=nothing, offset=2, title = nothing, plot_options...)
    display(loss)
    plt = plot()
    isnothing(scenario_nums) ? scenario_nums = rand(1:pbt.N_samples) : nothing
    if length(scenario_nums) == 1
        dd_sys, dd_spec = solve_sys_spec_n(pbt, scenario_nums[1], p)
        plt = plot(dd_sys, vars=1; label="System output", plot_options...)
        plot!(plt, dd_spec, vars=1; label="Specification output", plot_options...)
        plot!(plt, dd_spec.t, pbt.input_sample[scenario_nums[1]]; c=:gray, alpha=0.75, label="Input", plot_options...)
        title!("Input scenario $(scenario_nums[1])")
        display(plt)
    else
        j = 1
        if offset == -1
            plt = plot(layout = (length(scenario_nums),1))
            for i in scenario_nums
                dd_sys, dd_spec = solve_sys_spec_n(pbt, i, p)
                plot!(plt, dd_sys, vars=1; label = "System output (scenario $i)", c = palette(:tab20)[2*i-1], subplot = j, plot_options...)
                plot!(plt, dd_spec, vars=1; label = "Specification output (scenario $i))", c = palette(:tab20)[2*i], linestyle = :dash, subplot = j, plot_options...)
                plot!(plt, dd_spec.t, pbt.input_sample[i]; c=:gray, alpha=0.5, label = "Input $i", subplot = j, title = ("Input scenario $i"), plot_options...)
                #title!("Input scenarios $i")
                j+=1
          end
          xlabel!("t")
          ylabel!("output")
          display(plt)
        else
            for i in scenario_nums
                dd_sys, dd_spec = solve_sys_spec_n(pbt, i, p)
                plot!(dd_sys.t, dd_sys[1,:] .+ offset * (j - 1), vars=1; label=false, color_palette=:tab20, plot_options...) # yaxis = nothing "System output (scnenario $i)"
                plot!(plt, dd_spec.t, dd_spec[1,:] .+ offset * (j - 1), vars=1; label=false, linestyle=:dash, plot_options...) # "Specification output (scnenario $i))"
                j += 1
            end
            plot!([0.,0.01], [0.,0.];label="System output", c=:gray)
            plot!([0.,0.01], [0.,0.];label="Specification output", c=:gray, linestyle=:dash)
            scenarios_line = ""
            for s in @view scenario_nums[1:end - 1]
                scenarios_line *= "$s, "
            end
            scenarios_line *= "$(scenario_nums[end])"
            if isnothing(title)
                title!("Scenarios " * scenarios_line)
            else 
                title!(title)
            end
            xlabel!("t")
            ylabel!("output")
            display(plt)
        end
      
    end
    if !isnothing(fig_name)
          savefig(fig_name)
    end
    if !isnothing(loss_array)
          append!(loss_array, loss)
    end
      # Tell sciml_train to not halt the optimization.
      # If return true, then optimization stops.
    return false
end

#=
export plot_sys_graph
"""
Plot graph representations of optimized system and spec system.
"""
function plot_sys_graph(pbt::PBTProblem)
    nodecolor = [colorant"orange", colorant"lightseagreen"]
    nodefillc_sys = nodecolor[[1 for i in 1:pbt.size_p_sys]];
    nodefillc_spec = nodecolor[[1 for i in 1:pbt.size_p_spec]];
    nodefillc_sys[1] = nodecolor[2]
    nodefillc_spec[1] = nodecolor[2]
    L_sys = pbt.f_sys.L
    L_spec = pbt.f_spec.L
    g1 = gplot(SimpleGraph(L_sys), nodefillc=nodefillc_sys)
    g2 = gplot(SimpleGraph(L_spec), nodefillc=nodefillc_spec)
    display(g1)
    display(g2)
end
=#