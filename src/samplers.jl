export AbstractInputSampler, ContinousInputSampler, ConstantInputSampler, resample!, reset!

"""
    abstract type AbstractInputSampler end

Abstract type for all `PBTSamplers`. Each concret type musst implement

    `Base.getindex(sampler::ConcreteSamplerType, i)`

to get the i-th input sample.
"""
abstract type AbstractInputSampler end

"""
    resample!(s::AbstractInputSampler)

Clear the cache of the sampler. This will result in new samples.
"""
resample!(s::AbstractInputSampler) = empty!(s._cache)

"""
    reset!(s::AbstractInputSampler; seed=1)

Reset the RNG of the input sampler and clear cache.
"""
function reset!(s::AbstractInputSampler; seed=1)
    resample!(s)
    s.rng = MersenneTwister(seed)
end

"""
    ContinousInputSampler{SF} <: AbstractInputSampler

Input sampler to create time continous inputs of the form `(t) -> input`.

    ContinousInputSampler(f, t_span; seed=1)

Create an input sampler based on sample function f.
- `f`: function with signature `(rng, tspan, i)` which returns a anonymous function
  `(t) -> input` which represents the input.
- `tspan`: can be used inside `f` to determine the input
- `seed`: optional seed the RNG
"""
mutable struct ContinousInputSampler{SF} <: AbstractInputSampler
    rng::MersenneTwister
    samplef::SF
    tspan::Tuple{Float64, Float64}
    _cache::Vector{Any}

    ContinousInputSampler(f, t_span; seed=1) = new{typeof(f)}(MersenneTwister(seed), f, t_span, Any[])
end

function Base.getindex(s::ContinousInputSampler, idx::Int)
    clength = lastindex(s._cache)
    if idx > clength
        resize!(s._cache, idx)
        for i in clength+1:idx
            s._cache[i] = s.samplef(s.rng, s.tspan, i)
        end
    end
    return s._cache[idx]
end

"""
    ConstantInputSampler{SF} <: AbstractInputSampler

Input sampler to create constant inputs.

    ConstantInputSampler(f; seed=1)

Create an input sampler based on sample function f.
- `f`: function with signature `(rng, i)` which returns the input
- `seed`: optional seed the RNG
"""
mutable struct ConstantInputSampler{SF} <: AbstractInputSampler
    rng::MersenneTwister
    samplef::SF
    _cache::Vector{Any}

    ConstantInputSampler(f; seed=1) = new{typeof(f)}(MersenneTwister(seed), f, Any[])
end

function Base.getindex(s::ConstantInputSampler, idx::Int)
    clength = lastindex(s._cache)
    if idx > clength
        resize!(s._cache, idx)
        for i in clength+1:idx
            s._cache[i] = s.samplef(s.rng, i)
        end
    end
    return s._cache[idx]
end

Base.getindex(s::AbstractInputSampler, idxs::UnitRange{Int}) = [s[i] for i in idxs]
