cd(@__DIR__)
using Pkg
Pkg.activate(".")

##

using Revise
using Statistics
using ProBeTune

##
include("PlotUtils.jl")
##

t_steps = 0.:0.1:10.
tspan = (0., 10.)
N_osc = 10
size_p_sys = N_osc * (N_osc - 1) ÷ 2 + N_osc
size_p_spec = 2
N_samples = 10
p_sys_init = 6. * rand(size_p_sys) .+ 1.
p_spec_init = rand(size_p_spec) .+ 1.

p_initial = vcat(p_sys_init, repeat(p_spec_init, N_samples))

K_av = 1.

omega = randn(N_osc)
omega .-= mean(omega)

##

kur = PBTExample.create_kuramoto_example(omega, N_osc, size_p_spec, K_av, t_steps, N_samples)

##
plot_callback(kur, p_initial, 1.)

##
d, p_dist, = behavioural_distance(kur, p_initial; abstol=1e-4, reltol=1e-4)
d, p_dist, = behavioural_distance(kur, p_dist; abstol=1e-4, reltol=1e-4)
println("\nInitial distance to specified behaviour is lower than $d")

##
res = pbt_tuning(kur, p_dist; abstol=1e-4, reltol=1e-4)

res = pbt_tuning(kur, p_dist; abstol=1e-4, reltol=1e-4, optimizer_options = (:maxiters => 10, :cb => (p, l) ->  plot_callback(kur, p, l)))
p_tuned = res.minimizer

##
d2, p_dist_2, = behavioural_distance(kur, p_tuned; abstol=1e-4, reltol=1e-4)