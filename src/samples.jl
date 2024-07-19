function my_norm(θ; dims=ndims(θ), ϵ=eps(eltype(θ)), m=mean(θ, dims=dims), s=std(θ, dims=dims), ms=false)
    return ms ? ((θ .- m) ./ (s .+ ϵ), m, s) : (θ .- m) ./ (s .+ ϵ)
end

function my_unnorm(x, m, s; ϵ=eps(s))
    return x .* (s .+ ϵ) .+ m
end