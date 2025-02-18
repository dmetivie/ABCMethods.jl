"""
    ABC_SMC(N, dist_prior, simulate_and_distance, dist_K; factor=10, steps=15, minNeff = 3)
Perform the ABC-SMC algorithm. 
- `N` particles (or less in some extreme cases) are selected at each `step`
- `dist_prior`
Inpired by the [SimBIID](https://cran.r-project.org/web/packages/SimBIID/index.html).
## MA2 Example
```julia
dist_K(θ, arg...) = θ + randn(length(θ)) # simple perturbation
dist_K(θ, priorSample, priorLogW) = MvNormal(θ, cov(priorSample, StatsBase.weights(exp.(priorLogW), 2)))
m𝕃2(x, y) = mean(abs2, x - y)
ss_obs = [τ₁(y_obs), τ₂(y_obs)]
function simulate_and_distance(θ)
    y_sim = sample_from_(θ)
    ss = [τ₁(y_sim), τ₂(y_sim)]
    return m𝕃2(ss, ss_obs)
end
```
"""
function ABC_SMC(priorSample::AbstractMatrix, priorLogW::AbstractVector, N, dist_prior, simulate_and_distance, dist_K; steps=10, method=:classic, kwargs...)
    if method == :classic
        for t in 1:steps
            priorSample, priorLogW = abc_smc_step(N, dist_prior, priorSample, priorLogW, simulate_and_distance, dist_K; kwargs...)
        end
    elseif method == :force_success
        for t in 1:steps
            priorSample, priorLogW = abc_smc_step_force_success(N, dist_prior, priorSample, priorLogW, simulate_and_distance, dist_K; kwargs...)
        end
    end
    return priorSample[:, sample(1:length(priorLogW), Weights(exp.(priorLogW)), N)]
end

function ABC_SMC(N, dist_prior, simulate_and_distance, dist_K; kwargs...)
    priorLogW = log.(fill(1 / N, N))
    priorSample = rand(dist_prior, N)

    return ABC_SMC(priorSample, priorLogW, N, dist_prior, simulate_and_distance, dist_K; kwargs...)
end

#TODO: allow/force? in place version of `simulate_and_distance` to reduce memory usage for y_sim
#TODO: we use prepared_dist_𝐊 with functor to compute typically only onces a covariance matrix. We could use @memoize? or something a bit simpler for user (especially in simple cases) where there are no precomputation this is overkill


"""
    abc_smc_step(N, dist_prior, priorSample::AbstractMatrix, priorLogW::AbstractVector, simulate_and_distance, dist_K; factor=10)
Example
```julia
# Pre compute (a bit verbose) https://discourse.julialang.org/t/precompute-some-values-of-a-function/82079/7?u=dmetivie
struct distK{R}
    Σ::R
    distK(v,w) = new{typeof(v)}(cov(v,  StatsBase.weights(exp.(w)), 2))
end
(d::distK)(θ) = MvNormal(θ, d.Σ)

m𝕃2(x, y) = mean(abs2, x - y)
ss_obs = [τ₁(y_obs), τ₂(y_obs)]
function simulate_distance(θ)
    y_sim = sample_from_θ(θ)
    ss = [τ₁(y_sim), τ₂(y_sim)]
    return m𝕃2(ss, ss_obs)
end
```
"""
function abc_smc_step(N, dist_prior, priorSample::AbstractMatrix, priorLogW::AbstractVector, simulate_and_distance, dist_K; factor=10, minNeff=3)
    Neff = size(priorSample, 2) # Neff can be different from N in some cases 

    prepared_dist_𝐊 = dist_K(priorSample, priorLogW)

    # Resample based on weights
    rw = exp.(priorLogW)
    θstar = priorSample[:, sample(1:Neff, Weights(rw), N * factor)]

    # Propose new parameters
    # force that is inside the prior
    prop = map(eachcol(θstar)) do θ
        inprior = false
        t = similar(θ)
        while !inprior
            t .= rand(prepared_dist_𝐊(θ))
            inprior = insupport(dist_prior, t)
        end
        return t
    end

    # Compute distances
    distances = simulate_and_distance.(prop)
    qCut = quantile(distances, 1 / factor)

    # Filter proposals
    θstarstar = prop[distances.<qCut]
    # If extinction (or too few valid sample) -> resample from prior
    #! 2 samples is the minimum but lead to posdev issues with MvNormal kernel
    if length(θstarstar) < minNeff
        Nnew = N * 1000
        @warn "Only $(length(θstarstar)) valid samples -> resampling from original prior Nnew = N*1000 = $Nnew"
        return rand(dist_prior, Nnew), log.(fill(1 / Nnew, Nnew))
    end
    # Compute log weights
    lw = [compute_lw(θ, priorSample, priorLogW, dist_prior, prepared_dist_𝐊) for θ in θstarstar]

    # Normalize log weights
    mx = maximum(lw)
    lw .-= mx # remove max
    lw .-= logsumexp(lw) # normalize

    return reduce(hcat, θstarstar), lw
end

# Helper function to compute log weights
function compute_lw(θ, priorSample::AbstractMatrix, priorLogW::AbstractVector, dist_prior, dist_K)
    terms = priorLogW .+ [logpdf(dist_K(x), θ) for x in eachcol(priorSample)]
    mt = maximum(terms)
    denom = mt + logsumexp(terms .- mt)
    return logpdf(dist_prior, θ) - denom
end

function randin(d, prior)
    t = rand(d)
    inprior = insupport(prior, t)
    while !inprior
        t = rand(d)
        inprior = insupport(prior, t)
    end
    return t
end

function abc_smc_step_force_success(N, dist_prior, priorSample::AbstractMatrix, priorLogW::AbstractVector, simulate_and_distance, dist_K; factor=10, N_try_max=10000)
    Neff = size(priorSample, 2) # Neff can be different from N in some cases 

    prepared_dist_𝐊 = dist_K(priorSample, priorLogW)

    # Resample based on weights
    distances = similar(priorLogW, N * factor)
    prop = fill(priorSample[:, 1], N * factor)
    rw = exp.(priorLogW)
    for n in 1:(N*factor)
        status = :start
        N_try = 1
        t = priorSample[:, 1]
        distance = distances[1]
        while status != :success
            θstar = priorSample[:, sample(1:Neff, Weights(rw))]
            t .= randin(prepared_dist_𝐊(θstar), dist_prior)
            distance, status = simulate_and_distance(t)
            N_try += 1
            if N_try > N_try_max
                @warn "Did not produce one sucessful simulation after $(N_try-1) try. Last return code = $(status)"
                distance = Inf
                status = :success # fake success to break out of the loop
            end
        end
        prop[n] = t
        distances[n] = distance
        # @show distance, t, insupport(dist_prior, t)
    end

    qCut = quantile(distances, 1 / factor)

    # Filter proposals
    θstarstar = prop[distances.<qCut]
    # Compute log weights
    lw = [compute_lw(θ, priorSample, priorLogW, dist_prior, prepared_dist_𝐊) for θ in θstarstar]

    # Normalize log weights
    mx = maximum(lw)
    lw .-= mx # remove max
    lw .-= logsumexp(lw) # normalize
    # nlw = log.(exp.(lw) ./ sum(exp.(lw)))

    return reduce(hcat, θstarstar), lw
end