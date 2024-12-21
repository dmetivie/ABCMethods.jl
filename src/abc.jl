abstract type AbstractABC end

"""
    ABC_Nearestneighbours{T,F,Q}
ABC method with summary statistics
"""
struct ABC_Nearestneighbours{T,F,Q} <: AbstractABC
    α::T # portion of neighbours
    η::Q # Sumary Stats
    ∇::F # distance used
end

"""
    ABC_Nearestneighbours{T,F,Q}
ABC method with 
    α (Real) # portion of neighbours
    η (Function) # Sumary Stats
    ∇ (Function) # distance used
If η is not provided, the 𝕃2 norm is used as "summary statistic".
"""
ABC_Nearestneighbours(α, Δ) = ABC_NearestneighboursL2(α, Δ)

struct ABC_NearestneighboursL2{T,F} <: AbstractABC
    α::T # portion of neighbours
    ∇::F # distance used
end

"""

Select the `K×α` closest samples from `y` in `ys_sample`. Returns the associated `θ_sample`.

`dims`: let you choose which dimension samples are concatenated.
`all`: if `true`, consider that `η` applies to all sample at once (relevant for neural network `η`). 
if `false` apply to each sample separetly.
"""
function ABC_selection(y::AbstractArray, ys_sample::AbstractArray, θ_sample, abc::ABC_NearestneighboursL2; dims=ndims(ys_sample))
    N = size(ys_sample, dims) # last dims with samples
    @assert ndims(y) == ndims(ys_sample) - 1

    K = ceil(Int, N * abc.α)

    distances = [sum(abs2, y - y_sample) for y_sample in eachslice(ys_sample, dims=dims)]

    best = sortperm(distances)[1:K]
    return θ_sample[:, best]
end

function ABC_selection(ys::AbstractArray{T,dim}, ys_sample::AbstractArray{T,dim}, θ_sample, abc::ABC_Nearestneighbours; dims=dim, all_samples=true) where {T,dim}
    if all_samples
        η_obs = abc.η(ys)
        η_samples = abc.η(ys_sample)
    else
        ηall(X) = reduce(hcat, [abc.η(x) for x in eachslice(X, dims=dims)])
        η_obs = ηall(ys)
        η_samples = ηall(ys_sample)
    end
    return ABC_selection(η_obs, η_samples, θ_sample, ABC_NearestneighboursL2(abc.α, abc.∇); dims=ndims(η_samples))
end

function ABC_selection(ys::AbstractArray{T,dim}, ys_sample::AbstractArray{T,dim}, θ_sample, abc::ABC_NearestneighboursL2; dims=dim) where {T,dim}
    return [ABC_selection(y, ys_sample, θ_sample, ABC_NearestneighboursL2(abc.α, abc.∇); dims=dims) for y in eachslice(ys, dims=dims)]
end

function ABC2df(results, θ_test; q_min=0.025, q_max=0.975)
    θs_hat = reduce(hcat, mean.(results, dims=2))
    qs_min = reduce(hcat, [[quantile(r, q_min) for r in eachrow(S)] for S in results])
    qs_max = reduce(hcat, [[quantile(r, q_max) for r in eachrow(S)] for S in results])
    return [DataFrame(hcat(θ_test[i, :], θs_hat[i, :], qs_min[i, :], qs_max[i, :]), ["θ", "θ_hat", "q_low", "q_high"]) for i in axes(θ_test, 1)]
end