cd(@__DIR__)
using Pkg
Pkg.activate(".")

##

using Revise
using Statistics
using ProBeTune
using DiffEqFlux

##

using Random
Random.seed!(1)

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
omega_spreads = [1., 5., 12., 15.]
N_omega = length(omega_spreads)
omega = randn(N_osc)
omega .-= mean(omega)

##

kur1 = PBTExample.create_kuramoto_example(omega, N_osc, size_p_spec, K_av, t_steps, N_samples)
kur2 = PBTExample.create_kuramoto_example(omega.*2., N_osc, size_p_spec, K_av, t_steps, N_samples)
kur5 = PBTExample.create_kuramoto_example(omega.*5., N_osc, size_p_spec, K_av, t_steps, N_samples)
kur10 = PBTExample.create_kuramoto_example(omega.*10., N_osc, size_p_spec, K_av, t_steps, N_samples)

kur_exs = [kur1,kur2,kur5,kur10]
d_final = ones(length(kur_exs))
i = 1
##
for kur in kur_exs
        
    d, p_dist, = behavioural_distance(kur, p_initial; abstol=1e-4, reltol=1e-4,
                                        optimizer=DiffEqFlux.BFGS(),
                                        optimizer_options=(
                                            :maxiters => 20, 
                                            :cb => PBTLibrary.basic_pbt_callback))
    println("Initial distance to specified behaviour is lower than $d")

    ##

    res = pbt_tuning(kur, p_dist; abstol=1e-4, reltol=1e-4)
    p_tuned = res.minimizer
    ##
    scen = 1:5
    ##
    plot_callback(kur, p_tuned, res.minimum, scenario_nums = scen)
    ##
    d2, p_dist_2, = behavioural_distance(kur, p_tuned; abstol=1e-4, reltol=1e-4)

    println("After a first tuning using ADAM(0.01) the distance to specified behaviour is lower than $d2")

    ##

    res = pbt_tuning(kur, p_dist_2; abstol=1e-4, reltol=1e-4,
                    optimizer=DiffEqFlux.BFGS(),
                    optimizer_options=(
                        :maxiters => 5, 
                        :cb => PBTLibrary.basic_pbt_callback))
                    

    p_t2 = res.minimizer

    ##
    d3, p_dist_3, = behavioural_distance(kur, p_t2; abstol=1e-4, reltol=1e-4)

    println("After a second tuning using BFGS the distance to specified behaviour is lower than $d3")
    ##

    resample!(kur)
    d4, p_dist_4, = behavioural_distance(kur, p_dist_3; abstol=1e-4, reltol=1e-4,
                    optimizer = DiffEqFlux.BFGS(),
                    optimizer_options=(
                        :maxiters => 20, 
                        :cb => PBTLibrary.basic_pbt_callback))

    d4, p_dist_4, = behavioural_distance(kur, p_dist_4; abstol=1e-4, reltol=1e-4,
                    optimizer = DiffEqFlux.AMSGrad(0.01),
                    optimizer_options=(
                        :maxiters => 2000,))
        

    println("After resampling the distance to specified behaviour is lower than $d4")
    d_final[i] = d4
    i +=1
end