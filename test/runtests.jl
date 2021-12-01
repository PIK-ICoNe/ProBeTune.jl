cd(@__DIR__)
using Pkg
Pkg.activate(".")

##
using Test
using ProBeTune
using Statistics
using DiffEqFlux

# test samplers
@testset "Samplers" begin
    @testset "Continuos sampler" begin
        f(rng, t_span, n) = t -> rand(rng, 1)

        s = ContinousInputSampler(f, (0, 10))
        @test s.tspan == (0, 10)

        A = [f(0.0) for f in s[1:10]]
        resample!(s)
        B = [f(0.0) for f in s[1:10]]
        @test A !== B

        reset!(s)
        C = [f(0.0) for f in s[1:10]]
        @test A == C
    end

    @testset "Constant sampler" begin
        f(rng, n) =rand(rng, 5)

        s = ConstantInputSampler(f)

        A = s[1:10]
        resample!(s)
        B = s[1:10]
        @test A !== B

        reset!(s)
        C = s[1:10]
        @test A == C
    end
end

# Test a simple example
@testset "simple_example" begin
    dim_sys = 10
    diffusive_nl_example = PBTExample.create_diffusive_nl_example(dim_sys, 2, 0.:0.1:10., 10);

    p_initial = ones(2 * 10 + dim_sys)

    d_init, p_tuned, dists = behavioural_distance(diffusive_nl_example, p_initial; abstol=1e-2, reltol=1e-2)
    
    @test typeof(p_tuned) == typeof(p_initial)
    @test p_initial != p_tuned # test if the optimizer actually changes the parameters, this way we can also indirectly check if AD works

    res_10 = pbt_tuning(diffusive_nl_example, p_tuned; abstol=1e-4, reltol=1e-4, optimizer = ADAM(0.5), optimizer_options = (:maxiters => 50,))

    @test res_10.retcode == :Default || res_10.retcode == :Success

    d_end, p_tuned, dists = behavioural_distance(diffusive_nl_example, res_10.u; abstol=1e-2, reltol=1e-2)

    @test d_init > d_end # Testing if the tuning results in a smaller behavioural distance
end