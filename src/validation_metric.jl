"""
	NMAE(ŷ, y) = mean(abs, ŷ - y) / mean(abs, y)
Normalize Mean Absolute Error
"""
NMAE(ŷ, y) = mean(abs, ŷ - y) / mean(abs, y)

sd(ŷ, y) = std(abs.(ŷ - y))

function summary_method(df_result)
    @combine df_result begin
        :NMAE = NMAE(:θ_hat, :θ)
        :sd = sd(:θ_hat, :θ)
        :IC = mean(:q_high - :q_low)
        :MedIC = median(:q_high - :q_low)
        :Coverage = count(:q_low .≤ :θ .≤ :q_high)/length(:θ)            
    end
end

𝕃2(x, y) = sum(abs2, x - y)

"""
    inABCellipse(ABCresults, θ_test, α)
Using the ABC selected sample to estimate the posterior distribution, it computes the associated confidence ellispse, and Mahalanobis distance. It tests if this distance is bellow the `Chisq(dim)` `α`-quantile where `dim` is the dimension of the input vector.
WARNING: This assumes that the confidence region is Normally distributed.
"""
function inABCellipse(ABCresults::AbstractVector, θ_test::AbstractMatrix, thr)
    N = length(ABCresults)
    @assert N == size(θ_test, 2)
    ν = size(θ_test, 1) # dimension of the parameters
    InOrNot = zeros(Bool, N)
    for (i, x) in enumerate(ABCresults)
        m = mean(x, dims = 2)[:, 1]
        C = cov(x')
        ## squared Mahalanobis distance
        @views InOrNot[i] = dot(m - θ_test[:, i], inv(C), m - θ_test[:, i]) < thr
    end
    return count(InOrNot) / N
end

function compute_thresholds2D(x, α)
    kde_res = kde((x[1, :], x[2, :]))
    ik = InterpKDE(kde_res)
    ps = [pdf(ik, c[1], c[2]) for c in eachcol(x)]
    thresholds = hdr_thresholds(α, ps)
    return ik, thresholds
end

"""
    inHDR(ABCresults::AbstractVector, θ, α)
Compute the `α`-Highest Density Region (so far only in 2D) and verify is the test data is in.
"""
function inHDR(ABCresults::AbstractVector, θ_test, α)
    @assert length(ABCresults) == size(θ_test, 2)
    if size(θ_test, 1) == 2
        iks_thrs = compute_thresholds2D.(ABCresults, α)
        return count([pdf(iks_thrs[i][1], c[1], c[2]) > iks_thrs[i][2] for (i,c) in enumerate(eachcol(θ_test))])/length(ABCresults)
    else
        @warn "kde not yet implemented for $(size(result, 1))-Dimension"
    end
end

"""
    confidence_ellipse2D(x::AbstractVector, y::AbstractVector, s; n_points = 100)
    confidence_ellipse2D(X::AbstractMatrix, s; n_points = 100)
Create the points forming the `s`-confidence ellipse of `X` (or `x` and `y`).
To have a `α`-confidence region choose `s = quantile(Chi(2), α)` 
Can be used with `Plots.jl` and other plotting libraries.
```julia
using Plots
using Distributions
X = rand(MvNormal([2 0.5; 0.5 1]), 2000)
scatter(X[1,:], X[2,:])
α = quantile(Chi(2), α)
plot!(confidence_ellipse2D(X', α))
```
Idea from https://github.com/CarstenSchelp/CarstenSchelp.github.io/blob/master/LICENSE
"""
function confidence_ellipse2D(x::AbstractVector, y::AbstractVector, s; n_points = 100, cov_matrix = cov(hcat(x, y)))
    if length(x) != length(y)
        throw(ArgumentError("x and y must be the same size"))
    end

    
    pearson = cov_matrix[1, 2] / sqrt(cov_matrix[1, 1] * cov_matrix[2, 2])

    ell_radius_x = sqrt(1 + pearson)
    ell_radius_y = sqrt(1 - pearson)

    scale_x = sqrt(cov_matrix[1, 1]) * s
    mean_x = mean(x)

    scale_y = sqrt(cov_matrix[2, 2]) * s
    mean_y = mean(y)

    # Generate the ellipse points
    t = LinRange(0, 2π, n_points)
    ellipse_x = ell_radius_x * cos.(t)
    ellipse_y = ell_radius_y * sin.(t)

    # Rotation and scaling transformation
    rotation_matrix = [cos(π/4) -sin(π/4); 
                       sin(π/4)  cos(π/4)]
    rotated_ellipse = rotation_matrix * hcat(ellipse_x, ellipse_y)'
    scaled_ellipse = hcat(scale_x * rotated_ellipse[1, :], scale_y * rotated_ellipse[2, :])

    # # Translate the ellipse
    translated_ellipse_x = scaled_ellipse[:, 1] .+ mean_x
    translated_ellipse_y = scaled_ellipse[:, 2] .+ mean_y

    return (translated_ellipse_x, translated_ellipse_y)
end

confidence_ellipse2D(X::AbstractMatrix, s; n_points = 100, cov_matrix = cov(X)) = confidence_ellipse2D(X[:,1], X[:,2], s; n_points = n_points, cov_matrix = cov_matrix)

function areaellipse(X, s)
    cov_matrix = cov(X)
    ρ = cov_matrix[1, 2] / sqrt(cov_matrix[1, 1] * cov_matrix[2, 2])
    σ1 = sqrt(cov_matrix[1, 1])
    σ2 = sqrt(cov_matrix[2, 2])
    return sqrt(1-ρ^2)*σ1*σ2*s^2
end
function areaellipseΣ(cov_matrix, s)
    ρ = cov_matrix[1, 2] / sqrt(cov_matrix[1, 1] * cov_matrix[2, 2])
    σ1 = sqrt(cov_matrix[1, 1])
    σ2 = sqrt(cov_matrix[2, 2])
    return sqrt(1-ρ^2)*σ1*σ2*s^2
end

# areaellipseΣ(MC_testD[2][:, :, i],qhatD)
# aa = [areaellipse(results_posterior_cnn[i]',quantile(Chi(2), 0.95))
# for i in eachindex(results_posterior_cnn)]
# bb = [areaellipseΣ(MC_testD[2][:, :, i],qhatD) for i in eachindex(results_posterior_cnn)]

# sortperm(bb)
# findall(b[:,1] .== 0 .&& aa.>1.5bb .&& a.==1)