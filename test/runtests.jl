using Test
using ProBeTune

# test samplers
@testset "Samplers" begin
    @testset "Continous sampler" begin
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
