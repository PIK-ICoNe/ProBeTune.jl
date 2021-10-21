module ProBeTune

import Base.@kwdef

using OrdinaryDiffEq
using DiffEqFlux
using Statistics

export PBTProblem
"""
Struct defining a probabilistic behavioural tuning problem.

# Arguments
- `f_spec`: Right hand side function of the system, called as f(dy, y, i, p, t).
- `f_sys`: Right hand side function of the specification, called as f(dy, y, i, p, t).
- `input_distribution_sampler`: Sampler to generate inputs, this should provide inputs according to the distribution ρ, called as input_sampler(t_span, n).
- `output_metric`: Metric that calculates how far the outputs of system and spec are apart.
    Called as output_metric(sol_system, sol_spec), where sol is a DifferentialEquations solution object.
- `N_samples::Int`: Number of samples to use.
- `size_p_spec::Int`: Size of the parameter space of the spec.
- `size_p_sys::Int`: Size of the parameter space of the system.
- `y0_spec`: Initial conditions for integrating the spec.
- `y0_sys`: Initial conditions for integrating the system.
- `tsteps`: Times are which to save the solution
- `t_span`: Start and end time of the integration
- `solver = Tsit5()`: Solver for integrating spec and system.
- `input_sample = [input_sampler(t_span, n) for n in 1:N_samples]`: Field to save the current sample being worked on.
"""
@kwdef mutable struct PBTProblem
  f_spec
  f_sys
  input_distribution_sampler
  output_metric
  N_samples::Int
  size_p_spec::Int
  size_p_sys::Int
  y0_spec
  y0_sys
  tsteps
  t_span
  solver = Tsit5()
  input_sample = [input_distribution_sampler(t_span, n) for n in 1:N_samples]
end

export resample!
"""
Resample the inputs
"""
function resample!(pbt::PBTProblem; n = pbt.N_samples)
  pbt.N_samples = n
  pbt.input_sample = [pbt.input_distribution_sampler(pbt.t_span, n) for n in 1:pbt.N_samples]
end


###################
# Generating solutions
# The following functions generate solutions
###################

export solve_sys_spec
"""
Solve system and specification for given parameters
Parameters:
- `pbt::PBTProblem`
- `i`: input to the system
- `p_sys`: parameters of the system
- `p_spec`: parameters of specification
"""
function solve_sys_spec(pbt::PBTProblem, i, p_sys, p_spec; solver_options...)
    sol_sys = solve(ODEProblem((dy,y,p,t) -> pbt.f_sys(dy, y, i(t), p, t), pbt.y0_sys, pbt.t_span, p_sys), pbt.solver; saveat=pbt.tsteps, solver_options...)
    sol_spec = solve(ODEProblem((dy,y,p,t) -> pbt.f_spec(dy, y, i(t), p, t), pbt.y0_spec, pbt.t_span, p_spec), pbt.solver; saveat=pbt.tsteps, solver_options...)
    sol_sys, sol_spec
end

export solve_sys_spec_n
"""
Solve system and spec for the nth input sample given a stacked parameter array p.
Parameters:
- `pbt`: existing PBTProblem instance
- `n`: number of input in the array
- `p`: combined array of sys and spec parameters
"""
function solve_sys_spec_n(pbt::PBTProblem, n::Int, p; solver_options...)
    p_sys = view(p,1:pbt.size_p_sys)
    p_spec = view(p,(pbt.size_p_sys + 1 + (n - 1) * pbt.size_p_spec):(pbt.size_p_sys + n * pbt.size_p_spec))

    i = pbt.input_sample[n]

    solve_sys_spec(pbt, i, p_sys, p_spec; solver_options...)
end

###################
# Loss functions
# The following functions are loss functions actually used to
# tune the system and evaluate behavioural distances.
###################

export pbt_loss

"""
pbt_loss provides the loss function for the Probabilistic Tuning Problem.
"""
function pbt_loss(pbt::PBTProblem, p; solver_options...)
  loss = 0.

  for n in 1:pbt.N_samples
      loss += pbt.output_metric(solve_sys_spec_n(pbt, n, p; solver_options...)...)
  end

  loss / pbt.N_samples
end

"""
pbt_individual_loss provides the contribution to the loss function for the Probabilistic Tuning Problem for a single sample.
"""
function pbt_individual_loss(pbt::PBTProblem, n, p_sys, p_spec; solver_options...)
  # loss function for sample n only
  i = pbt.input_sample[n]
  pbt.output_metric(solve_sys_spec(pbt, i, p_sys, p_spec; solver_options...)...)
end

###################
# Behavioural Distance
# For a fixed system we can provide a given p_sys.
###################

export behavioural_distance

"""
Tune the specification parameters only to calculate the behvaioural distance of the system
to the specification.

# Arguments:
- `pbt`: PBT problem
- `p`: stacked array of system parameters and initial parameters for the specifications
- `optimizer`: choose optimization algorithm (default `DiffEqFlux.ADAM(0.01)`)
"""
function behavioural_distance(pbt::PBTProblem, p; verbose=true, optimizer=DiffEqFlux.ADAM(0.01), optimizer_options=(:maxiters => 1000,), solver_options...)
  # Optimize the specs only

  p_tuned = copy(p)

  @views begin
      p_sys = p_tuned[1:pbt.size_p_sys]
      p_specs = [p_tuned[pbt.size_p_sys + 1 + (n - 1) * pbt.size_p_spec:pbt.size_p_sys + n * pbt.size_p_spec] for n in 1:pbt.N_samples]
  end

  verbose && print("0 out of $(pbt.N_samples) samples evaluated")

  distances = zeros(pbt.N_samples)

  for n in 1:pbt.N_samples
    res = DiffEqFlux.sciml_train(
        x -> pbt_individual_loss(pbt, n, p_sys, x; solver_options...),
        Array(p_specs[n]),
        optimizer;
        optimizer_options...
        )
    verbose && print("\r$n out of $(pbt.N_samples) samples evaluated")
    p_specs[n] .= res.minimizer
    distances[n] = res.minimum
  end
  print("\n")
  mean(distances), p_tuned, distances
end


###################
# Tuning the system
###################

export pbt_tuning

"""
Tune the system to the specification.

# Arguments:
- `pbt`: PBT problem
- `p`: stacked array of initial system and specification parameters
- `optimizer`: choose optimization algorithm (default `DiffEqFlux.ADAM(0.01)`)
- `optimizer_options`: choose optimization options (default `(:maxiters => 100,)`)
- `solver_options...` all further options are passed through to the differential equations solver
"""
function pbt_tuning(pbt::PBTProblem, p; optimizer=DiffEqFlux.ADAM(0.01), optimizer_options=(:maxiters => 100,), solver_options...)
  DiffEqFlux.sciml_train(
    x -> pbt_loss(pbt, x; solver_options...),
    p,
    optimizer;
    optimizer_options...
    )
end

#############################
#############################
#############################
#############################
# Library of further things

export PBTLibrary
include("PBTLibrary.jl")

export PBTExample
include("PBTExample.jl")

end