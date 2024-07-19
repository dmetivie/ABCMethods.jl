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
"""
function ABC_selection(y, ys_sample, θ_sample, abc::ABC_NearestneighboursL2; dim = ndims(ys_sample))
	
    N = size(ys_sample, dim) # last dims with samples

    K = ceil(Int, N * abc.α)

    distances = [sum(abs2, y - y_sample) for y_sample in eachslice(ys_sample, dims = dim)]

    best = sortperm(distances)[1:K]
    return θ_sample[:, best]
end

function ABC_selection(ys, y_sample, θ_sample, abc::ABC_Nearestneighbours)
    N = size(y_sample)[end]
    n = size(ys)[end]

    K = ceil(Int, N * abc.α)

    η_obs = abc.η(ys)
    η_sampled = abc.η(y_sample)
    distances = [[sum(abs2, col_obs - col) for col in eachcol(η_sampled)] for col_obs in eachcol(η_obs)]

    best = [sortperm(distances[i])[1:K] for i in 1:n]
    return [θ_sample[:, best[i]] for i in 1:n]
end

