cd(@__DIR__)
using Pkg
Pkg.activate(".")

##

using Revise
using Statistics
using ProBeTune
using DiffEqFlux
using LaTeXStrings

##

using Random
Random.seed!(1)

##
include("PlotUtils.jl")

##

#=
## Simple graph example
As an example we want to tune a system of 10 nodes with one grid connection point to a specification of 2 nodes.
Our goal is to find such parameters for the 10-node system that it behaves close enough to a 2-node system under all possible inputs.
In this example we start with just 10 possible inputs.
=#
dim_sys = 10
graph_example = PBTExample.create_graph_example(dim_sys, 2, 0.:0.1:10., 10);

# => Can we make a large graph react like a small graph by tuning the non-linearity at the nodes?

p_initial = ones(2*10+dim_sys)
##
#=
Finished initialization
=#
d, p_tuned, dists = behavioural_distance(graph_example, p_initial; abstol=1e-2, reltol=1e-2)

##
scenarios = 1:3 # we can choose what scnearios to use for plots everywhere
plot_callback(graph_example, p_initial, d, scenario_nums=scenarios)

##
res_10 = pbt_tuning(graph_example, p_tuned; abstol=1e-4, reltol=1e-4,
                    optimizer = ADAM(0.5),
                    optimizer_options = (
                        :maxiters => 25,
                        :cb => PBTLibrary.basic_pbt_callback))

p_tuned = res_10.minimizer
##
res_10 = pbt_tuning(graph_example, p_tuned; abstol=1e-6, reltol=1e-6,
                    optimizer = ADAM(0.1),
                    optimizer_options = (
                        :maxiters => 50,
                        :cb => PBTLibrary.basic_pbt_callback))

p_tuned = res_10.minimizer
##
res_10 = pbt_tuning(graph_example, p_tuned; abstol=1e-6, reltol=1e-6,
                    optimizer = DiffEqFlux.BFGS(),
                    optimizer_options = (
                        :maxiters => 10,
                        :cb => PBTLibrary.basic_pbt_callback))

p_tuned = res_10.minimizer

plot_callback(graph_example, p_tuned, res_10.minimum, scenario_nums = scenarios)

##

#=
After getting a good initial approximation, we can look at the minimizer.
Going further makes little sense as we would be overfitting to the small sample.
=#
print(relu.(p_tuned))

#= ## Resampling
We can check the quality of the resulting minimizer by optimizing the specs only (a much simpler repeated 2d optimization problem)
=#
d, p2 = behavioural_distance(graph_example, p_tuned)

#=
In order to understand how much we were overfitting with respect to the concrete sample, we resample.
That is, we generate a new problem with a new sample from the same distribution:
This will give us information on the system tuning with a sample different from the one that the tuning was optimized for.
=#
resample!(graph_example)
d_rs, p3 = behavioural_distance(graph_example, p_tuned)

println(d_rs/d)

##
plot_callback(graph_example, p3, d_rs, scenario_nums = scenarios)
#=
The median individual loss has gone up by a factor of 4. 
This means that the system is somewhat overfit to the initial sample.
## Larger number of samples
We warmed up the optimization with a very small number of scenarios, we can now initialize a higher sampled optimization using the system parameters found in the lower one:
=#
p_100 = ones(2*100+dim_sys)
p_100[1:dim_sys] .= p2[1:dim_sys]
p_100[dim_sys+1:end] .= repeat(p2[dim_sys+1:dim_sys+2], 100)

resample!(graph_example; n = 100);

#=
Optimizing only the specs is a task linear in the number of scenarios, the idea is that this will help with warming up the optimization.
We can also study the quality of the tuning found by the optimization based on a small number of samples.
=#
d100, p_100_initial = behavioural_distance(graph_example, p_100;
                    optimizer = DiffEqFlux.ADAM(0.1),
                    optimizer_options = (:maxiters => 10,),
                    abstol = 1e-3, reltol=1e-3)

plot_callback(graph_example, p_100_initial, d100, scenario_nums = scenarios)

#= Now we can train the full system:
=#
res_100 = pbt_tuning(graph_example, p_100_initial; abstol=1e-4, reltol=1e-4,
                    optimizer = BFGS(),
                    optimizer_options = (
                        :maxiters => 10,
                        :cb => PBTLibrary.basic_pbt_callback))

p_tuned = res_100.minimizer

plot_callback(graph_example, p_tuned, res_100.minimum, scenario_nums = scenarios)
#= Continue improving it for 150 Steps with some plotting in between:=#
for i in 1:30
    global res_100
    res_100 = pbt_tuning(graph_example, p_100_initial; abstol=1e-6, reltol=1e-6,
                        optimizer = BFGS(),
                        optimizer_options = (
                            :maxiters => 10,
                            :cb => PBTLibrary.basic_pbt_callback))
    plot_callback(graph_example, res_100.minimizer, res_100.minimum, scenario_nums = scenarios)
end

plot_callback(graph_example, res_10.minimizer, res_10.minimum; scenario_nums=scenarios)

d, p, losses = behavioural_distance(graph_example, res_10.minimizer)

x= exp10.(range(log10(.05),stop=log10(0.4), length = 50))
plot(x, (x)->1-PBTLibrary.confidence_interval(losses, x)[1],
    xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false,c=:blue)
    #label="Fraction of scenarios within set distance from specification")
    plot!(x, (x)->1-PBTLibrary.confidence_interval(losses, x)[2],
        xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false, linestyle=:dash, c=:blue)
        #label="Fraction of scenarios within set distance from specification")
    plot!(x, (x)->1-PBTLibrary.confidence_interval(losses, x)[3],
        xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false, linestyle=:dash, c=:blue)
        #label="Fraction of scenarios within set distance from specification")

plot(sort!(losses),[1:length(losses)]./length(losses), label = false)
