module ProBeTune

import Base.@kwdef

using OrdinaryDiffEq
using DiffEqFlux
using Statistics
using Random

include("samplers.jl")

export PBTProblemParameters
"""
    PBTProblemParameters(; kw...)

## Keywords:
- `N_samples::Int`: Number of samples to use.
- `size_p_spec::Int`: Size of the parameter space of the spec.
- `size_p_sys::Int`: Size of the parameter space of the system.
- `output_metric`: Metric that calculates how far the outputs of system and spec are apart.
    Called as `output_metric(sol_system, sol_spec)`, where sol is a DifferentialEquations solution object.
"""
Base.@kwdef mutable struct PBTProblemParameters
    N_samples::Int
    size_p_spec::Int
    size_p_sys::Int
    output_metric
end

export AbstractPBTProblem
"""
    AbstractPBTProblem

Supertype type to define PBTProblems.

Each concrete subtype has to have the fields
- p::PBTProblemParameters
- input_sampler::AbstractInputSampler

Each concrete subtype has to implement

    solve_sys_spec(pbt::ConcretePBTProblem, input, p_sys, p_spec) -> sol_sys, sol_spec

which returns a tuple of two DiffEq solution objects: one for the sys, one for the spec.
"""
abstract type AbstractPBTProblem end


###################
# Definition of the default 'rhs' PBT problem
###################
export PBTProblem
"""
    PBTProblem <: AbstractPBTProblem

Characterize problems by their rhs.
"""
mutable struct PBTProblem <: AbstractPBTProblem
    p::PBTProblemParameters
    input_sampler::ContinousInputSampler
    f_spec
    f_sys
    y0_spec
    y0_sys
    tsteps
    t_span
    solver
end

"""
    PBTProblem(kw...)

Keyword constructor for default PBT Problem.

## Keywords:
- `N_samples::Int`: Number of samples to use.
- `size_p_spec::Int`: Size of the parameter space of the spec.
- `size_p_sys::Int`: Size of the parameter space of the system.
- `output_metric`: Metric that calculates how far the outputs of system and spec are apart.
    Called as `output_metric(sol_system, sol_spec)`, where sol is a DifferentialEquations solution object.
- `input_sampler`: `ContinousInputSampler`
- `f_spec`: Right hand side function of the system, called as f(dy, y, i, p, t).
- `f_sys`: Right hand side function of the specification, called as f(dy, y, i, p, t).
- `y0_spec`: Initial conditions for integrating the spec.
- `y0_sys`: Initial conditions for integrating the system.
- `tsteps`: Times are which to save the solution
- `t_span`: Start and end time of the integration
- `solver = Tsit5()`: Solver for integrating spec and system.
"""
function PBTProblem(; N_samples, size_p_spec, size_p_sys, output_metric,
                    input_sampler, f_spec, f_sys, y0_spec, y0_sys, tsteps, t_span, solver = Tsit5())
    p = PBTProblemParameters(; N_samples, size_p_spec, size_p_sys, output_metric)
    PBTProblem(p, input_sampler, f_spec, f_sys, y0_spec, y0_sys, tsteps, t_span, solver)
end

"""
    resample!(pbt::AbstractPBTProblem; n = pbt.p.N_samples)

Resample the inputs.
"""
function resample!(pbt::AbstractPBTProblem; n = pbt.p.N_samples)
    pbt.p.N_samples = n
    resample!(pbt.input_sampler)
end

###################
# Generating solutions
# The following functions generate solutions
###################

export solve_sys_spec
"""
    solve_sys_spec(pbt::PBTProblem, i, p_sys, p_spec; solver_options...)

Solve system and specification for given parameters `p_sys` and `p_spec` and input `i`.
Wraps the `f_sys` and `f_spec` rhs functions as an ODEProblem function. `solver_options` kw arguments get
passed to both solve calls.
"""
function solve_sys_spec(pbt::PBTProblem, i, p_sys, p_spec; solver_options...)
    sol_sys = solve(ODEProblem((dy,y,p,t) -> pbt.f_sys(dy, y, i(t), p, t), pbt.y0_sys, pbt.t_span, p_sys), pbt.solver; saveat=pbt.tsteps, solver_options...)
    sol_spec = solve(ODEProblem((dy,y,p,t) -> pbt.f_spec(dy, y, i(t), p, t), pbt.y0_spec, pbt.t_span, p_spec), pbt.solver; saveat=pbt.tsteps, solver_options...)
    sol_sys, sol_spec
end

export solve_sys_spec_n
"""
    solve_sys_spec_n(pbt::AbstractPBTProblem, n::Int, p; solver_options...)

Solve system and spec for the n-th input sample given a stacked parameter array p.
Parameters:
- `pbt`: existing AbstractPBTProblem instance
- `n`: number of input in the array
- `p`: combined array of sys and spec parameters
"""
function solve_sys_spec_n(pbt::AbstractPBTProblem, n::Int, p; solver_options...)
    p_sys = view(p,1:pbt.p.size_p_sys)
    p_spec = view(p,(pbt.p.size_p_sys + 1 + (n - 1) * pbt.p.size_p_spec):(pbt.p.size_p_sys + n * pbt.p.size_p_spec))

    i = pbt.input_sampler[n]

    solve_sys_spec(pbt, i, p_sys, p_spec; solver_options...)
end

###################
# Loss functions
# The following functions are loss functions actually used to
# tune the system and evaluate behavioural distances.
###################

export pbt_loss

"""
    pbt_loss(pbt::AbstractPBTProblem, p; solver_options...)

pbt_loss calculates the loss function for the Probabilistic Tuning Problem.
It will add up the loss provided by the `output_metric` for all samples.
"""
function pbt_loss(pbt::AbstractPBTProblem, p; solver_options...)
    loss = 0.

        for n in 1:pbt.p.N_samples
            sol_sys, sol_spec = solve_sys_spec_n(pbt, n, p; solver_options...)
            loss += pbt.p.output_metric(sol_sys, sol_spec)
        end

    loss / pbt.p.N_samples
end

"""
    pbt_individual_loss(pbt::AbstractPBTProblem, n, p_sys, p_spec; solver_options...)

pbt_individual_loss provides the contribution to the loss function for the Probabilistic Tuning Problem for a single sample.
"""
function pbt_individual_loss(pbt::AbstractPBTProblem, n, p_sys, p_spec; solver_options...)
  # loss function for sample n only
  i = pbt.input_sampler[n]
  pbt.p.output_metric(solve_sys_spec(pbt, i, p_sys, p_spec; solver_options...)...)
end

###################
# Behavioural Distance
# For a fixed system we can provide a given p_sys.
###################

export behavioural_distance

"""
    behavioural_distance(pbt::AbstractPBTProblem, p; verbose=true, optimizer=DiffEqFlux.ADAM(0.01),
                         optimizer_options=(:maxiters => 1000,), solver_options...)

Tune the specification parameters only to calculate the behvaioural distance of the system
to the specification.

# Arguments:
- `pbt`: PBT problem
- `p`: stacked array of system parameters and initial parameters for the specifications
- `optimizer`: choose optimization algorithm (default `DiffEqFlux.ADAM(0.01)`)
"""
function behavioural_distance(pbt::AbstractPBTProblem, p; verbose=true, optimizer=DiffEqFlux.ADAM(0.01),
                              optimizer_options=(:maxiters => 1000,), solver_options...)
  # Optimize the specs only

  p_tuned = copy(p)

  @views begin
      p_sys = p_tuned[1:pbt.p.size_p_sys]
      p_specs = [p_tuned[pbt.p.size_p_sys + 1 + (n - 1) * pbt.p.size_p_spec:pbt.p.size_p_sys + n * pbt.p.size_p_spec] for n in 1:pbt.p.N_samples]
  end

  verbose && print("0 out of $(pbt.p.N_samples) samples evaluated")

  distances = zeros(pbt.p.N_samples)

  for n in 1:pbt.p.N_samples
    res = DiffEqFlux.sciml_train(
        x -> pbt_individual_loss(pbt, n, p_sys, x; solver_options...),
        Array(p_specs[n]),
        optimizer;
        optimizer_options...
        )
    verbose && print("\r$n out of $(pbt.p.N_samples) samples evaluated")
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
    pbt_tuning(pbt::AbstractPBTProblem, p; optimizer=DiffEqFlux.ADAM(0.01), optimizer_options=(:maxiters => 100,), solver_options...)

Tune the system to the specification.

# Arguments:
- `pbt`: PBT problem
- `p`: stacked array of initial system and specification parameters
- `optimizer`: choose optimization algorithm (default `DiffEqFlux.ADAM(0.01)`)
- `optimizer_options`: choose optimization options (default `(:maxiters => 100,)`)
- `solver_options...` all further options are passed through to the differential equations solver
"""
function pbt_tuning(pbt::AbstractPBTProblem, p; optimizer=DiffEqFlux.ADAM(0.01), optimizer_options=(:maxiters => 100,), solver_options...)
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
