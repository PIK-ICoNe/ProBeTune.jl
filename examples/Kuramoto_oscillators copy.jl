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
##

omega = randn(N_osc)
omega[1] = 0.
omega .-= mean(omega[1:end])

##

function tune_run(omega, scale; N_osc = 10, size_p_spec = 2, K_av = 1., t_steps = 0.:0.1:10., N_samples = 10, save_plots = false)
    kur = PBTExample.create_kuramoto_example(omega*scale, N_osc, size_p_spec, K_av, t_steps, N_samples)

    d, p_dist, = behavioural_distance(kur, p_initial; abstol=1e-4, reltol=1e-4,
                                        optimizer=DiffEqFlux.BFGS(),
                                        optimizer_options=(
                                            :maxiters => 5, 
                                            :cb => PBTLibrary.basic_pbt_callback))
    d_initial = d
    println("Initial distance to specified behaviour is lower than $d")

    ##

    res = pbt_tuning(kur, p_dist; abstol=1e-4, reltol=1e-4)
    p_tuned = res.minimizer
    ##
    scen = 1:5
    ##
    plot_callback(kur, p_tuned, res.minimum, scenario_nums = scen)
    if save_plots
        savefig("../plots/res_"*string(round(scale, digits = 4))*"_initial_d$d.png")
    end
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
                        :maxiters => 10, 
                        :cb => PBTLibrary.basic_pbt_callback))

    d5, p_dist_5, = behavioural_distance(kur, p_dist_4; abstol=1e-4, reltol=1e-4,
                    optimizer = DiffEqFlux.AMSGrad(0.01),
                    optimizer_options=(
                        :maxiters => 2000,))
        
    plot_callback(kur, p_dist_5, d5, scenario_nums = scen)
    if save_plots
        savefig("../plots/res_"*string(round(scale, digits = 4))*"_final_d$d5.png")
    end
    println("After resampling the distance to specified behaviour is lower than $d4")
    d_final = d5
    d_initial, d_final
end
##
#res = [tune_run(kur) for kur in kur_exs]
scales = [1., 2., 3., 4., 5., 6.]
@time res1 = tune_run(omega, 1., save_plots = true)


res2 = tune_run(omega, 2.)
res5 = tune_run(omega, 5.)
res7 = tune_run(omega, 7.)
@time res5 = tune_run(omega, 5.)
@time res6 = tune_run(omega, 6.)

##
res = [res1, res2, res5]

q = [r[1]/r[2] for r in res]
d, p_dist, = behavioural_distance(kur7, p_initial; abstol=1e-4, reltol=1e-4,
optimizer=DiffEqFlux.BFGS(),
optimizer_options=(
    :maxiters => 10, 
    :cb => PBTLibrary.basic_pbt_callback))
res0 = pbt_tuning(kur7, p_dist; abstol=1e-4, reltol=1e-4, optimizer_options = (:maxiters => 200))
