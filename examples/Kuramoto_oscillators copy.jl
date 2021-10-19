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
path = "../plots/kuramoto/"
##
global t_steps = 0.:0.1:10.
global tspan = (0., 10.)
global N_osc = 10
global size_p_sys = N_osc * (N_osc - 1) ÷ 2 + N_osc
global size_p_spec = 2
global N_samples = 10
global p_sys_init = 6. * rand(size_p_sys) .+ 1.
global p_spec_init = rand(size_p_spec) .+ 1.
global p_initial = vcat(p_sys_init, repeat(p_spec_init, N_samples))
global K_av = 1.
##
omega = randn(N_osc)
omega[1] = 0.
omega .-= mean(omega[1:end])
##
function tune_run(omega, scale; save_plots = false)
    Random.seed!(1)
    kur = PBTExample.create_kuramoto_example(omega*scale, N_osc, size_p_spec, K_av, t_steps, N_samples)

    d, p_dist, = behavioural_distance(kur, p_initial; abstol=1e-4, reltol=1e-4,
                                        optimizer=DiffEqFlux.BFGS(),
                                        optimizer_options=(
                                            :maxiters => 10, 
                                            :cb => PBTLibrary.basic_pbt_callback))
    d_initial = d
    println("Initial distance to specified behaviour is lower than $d\n")
    ##
    scen = 1:3
    ##
    plot_callback(kur, p_dist, d, scenario_nums = scen, title = "Scenarios $scen, s = $scale", offset = 0.5, xlims = (kur.t_span[2]/2, kur.t_span[2]))
    if save_plots
        savefig(path*string(round(scale, digits = 4))*"_initial_"*string(round(d, digits = 4))*".png")
    end
    ##
    res = pbt_tuning(kur, p_dist; abstol=1e-4, reltol=1e-4)
    p_tuned = res.minimizer
    ##
    d2, p_dist_2, = behavioural_distance(kur, p_tuned; abstol=1e-4, reltol=1e-4)

    println("After a first tuning using ADAM(0.01) the distance to specified behaviour is lower than $d2\n")

    ##

    res = pbt_tuning(kur, p_dist_2; abstol=1e-4, reltol=1e-4,
                    optimizer=DiffEqFlux.BFGS(),
                    optimizer_options=(
                        :maxiters => 5, 
                        :cb => PBTLibrary.basic_pbt_callback))
    
    p_t2 = res.minimizer
    ##
    plot_callback(kur, p_t2, res.minimum, scenario_nums = scen, title = "Scenarios $scen, s = $scale", offset = 0.5, xlims = (kur.t_span[2]/2, kur.t_span[2]))

    if save_plots
        savefig(path*string(round(scale, digits = 4))*"_tuned_"*string(round(res.minimum, digits = 4))*".png")
    end
    ##
    d3, p_dist_3, = behavioural_distance(kur, p_t2; abstol=1e-4, reltol=1e-4, optimizer_options=(:maxiters => 1000,))

    println("After a second tuning using BFGS the distance to specified behaviour is lower than $d3\n")
    ##
    plot_callback(kur, p_dist_3, d3, scenario_nums = scen, title = "Scenarios $scen, s = $scale", offset = 0.5, xlims = (kur.t_span[2]/2, kur.t_span[2]))
    if save_plots
        savefig(path*string(round(scale, digits = 4))*"_tuned2_"*string(round(d3, digits = 4))*".png")
    end
    resample!(kur)
    d4, p_dist_4, = behavioural_distance(kur, p_dist_3; abstol=1e-4, reltol=1e-4,
                    optimizer = DiffEqFlux.BFGS(),
                    optimizer_options=(
                        :maxiters => 10, 
                        :cb => PBTLibrary.basic_pbt_callback))

    d5, p_dist_5, = behavioural_distance(kur, p_dist_4; abstol=1e-4, reltol=1e-4,
                    optimizer = DiffEqFlux.AMSGrad(0.01),
                    optimizer_options=(
                        :maxiters => 1000,))
    
    plot_callback(kur, p_dist_5, d5, scenario_nums = scen, title = "Scenarios $scen, s = $scale", offset = 0.5, xlims = (kur.t_span[2]/2, kur.t_span[2]))
    if save_plots
        savefig(path*string(round(scale, digits = 4))*"_final_rs_"*string(round(d5, digits = 4))*".png")
    end
    println("After resampling the distance to specified behaviour is lower than $d4\n")
    d_final = d5
    p_final = p_dist_5
    d_initial, d_final, p_final
end
##

scales = vcat(1.0:0.1:1.5,2.0:0.5:5.)

##

res = [tune_run(omega, s, save_plots = true) for s in scales]

##
q = [r[1]/r[2] for r in res]

plot(scales, q)
savefig(path*"improvement(scale).png")
#scales2 = 6.:19.
#res2 = [tune_run(omega, s, save_plots = true) for s in scales2]

plot(scales, [r[2] for r in res], xlabel = "s", ylabel = "d", label = "Final distance to specification")
savefig(path*"final_d(omega).png")

## Continue optimization for s=1.2
kur_new = PBTExample.create_kuramoto_example(omega*1.2, N_osc, size_p_spec, K_av, t_steps, N_samples)
p_initial_new = res[3][3] # for scale = 1.2

d, p_dist, = behavioural_distance(kur_new, p_initial_new; abstol=1e-4, reltol=1e-4,
                    optimizer = DiffEqFlux.BFGS(),
                    optimizer_options=(
                        :maxiters => 10, 
                        :cb => PBTLibrary.basic_pbt_callback))
##
scen = 1:3
plot_callback(kur_new, p_dist, d, scenario_nums = scen, title = "Scenarios $scen, s = 1.2", offset = 0.5, xlims = (kur_new.t_span[2]/2, kur_new.t_span[2]))


##
d, p_dist, = behavioural_distance(kur_new, p_initial_new; abstol=1e-4, reltol=1e-4,
                    optimizer = DiffEqFlux.BFGS(),
                    optimizer_options=(
                        :maxiters => 10, 
                        :cb => PBTLibrary.basic_pbt_callback))

plot_callback(kur_new, p_dist, d, scenario_nums = scen, title = "Scenarios $scen, s = 1.2", offset = 0.5, xlims = (kur_new.t_span[2]/2, kur_new.t_span[2]))
##
res = pbt_tuning(kur_new, p_dist)

plot_callback(kur_new, res.minimizer, res.minimum, scenario_nums = scen, title = "Scenarios $scen, s = 1.2", offset = 0.5, xlims = (kur_new.t_span[2]/2, kur_new.t_span[2]))

savefig(path*"1.2_tuned_3_"*string(round(res.minimum, digits = 4))*".png")

resample!(kur_new)

d_rs, p_dist_rs = behavioural_distance(kur_new, res.minimizer; abstol=1e-4, reltol=1e-4,
optimizer = DiffEqFlux.BFGS(),
optimizer_options=(
    :maxiters => 10, 
    :cb => PBTLibrary.basic_pbt_callback))

plot_callback(kur_new, p_dist_rs, d_rs, scenario_nums = scen, title = "Scenarios $scen, s = 1.2", offset = 0.5, xlims = (kur_new.t_span[2]/2, kur_new.t_span[2]))

savefig(path*"1.2_rs_further_"*string(round(d_rs, digits = 4))*".png")
##
res2 = pbt_tuning(kur_new, p_dist_rs)

plot_callback(kur_new, res2.minimizer, res2.minimum, scenario_nums = scen, title = "Scenarios $scen, s = 1.2", offset = 0.5, xlims = (kur_new.t_span[2]/2, kur_new.t_span[2]))

savefig(path*"1.2_rtuned4_"*string(round(res2.minimum, digits = 4))*".png")
##
resample!(kur_new)

d_rs, p_dist_rs = behavioural_distance(kur_new, res.minimizer; abstol=1e-4, reltol=1e-4,
optimizer = DiffEqFlux.BFGS(),
optimizer_options=(
    :maxiters => 10, 
    :cb => PBTLibrary.basic_pbt_callback))

plot_callback(kur_new, p_dist_rs, d_rs, scenario_nums = scen, title = "Scenarios $scen, s = 1.2", offset = 0.5, xlims = (kur_new.t_span[2]/2, kur_new.t_span[2]))

savefig(path*"1.2_rs_final_"*string(round(d_rs, digits = 4))*".png")