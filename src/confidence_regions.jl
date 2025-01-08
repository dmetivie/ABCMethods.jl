"""
A circle/sphere of radius `radius` is definied by the set of point `x` such that `x' Σ x ≤ radius^2`
# Arguments
- `center` vector of the ellipsoid.
- `Σ`: Symmetric matrix (typically a covariance matrix).
- `radius`: Radius of the ellipsoid.
- `N`: Number of angles used in the ellipse.

# Returns
A tuple containing the (x,y[,z]) coordinates of the points forming the ellipse.

Code inspired from the [gellipsoid R package](https://github.com/friendly/gellipsoid/blob/master/R/ellipsoid.R).

# Example 
## 2D ellipse of confidence
```julia
using Distributions, Plots
malo_dist(x, Σ, s) = dot(x, inv(Σ), x) < s^2
Σ = [2 0.9; 0.9 0.5]
X = rand(MvNormal(Σ), 20000)
α = 0.95
s = quantile(Chi(2), α)
scatter(X[1,:], X[2,:], aspect_ratio = :equal, c = 1 .+ 2*[malo_dist(x, Σ, s) for x in eachcol(X)], label = :none)
plot!(ellipsoid([0,0], Σ, s; segments = 150), label = "ellipse of 95% confidence", lw = 2)

# Show eigenvectors
eg = eigen(inv(Σ))
λ = eg.values
v = eg.vectors
[plot!(tuple([[0, s*λ[i]^(-1/2)*c[j]] for j in axes(Σ,1)]...), label = :none, c=4,arrow=true, lw = 2) for (i,c) in enumerate(eachcol(v))]

xlabel!("x")
ylabel!("y")
ylims!(-5,5)
xlims!(-5,5)
```
## 3D ellipsoid of confidence
```julia
Σ = [1 0.9 0.1; 
    0.9 1 0.1;
    0.1 0.1 1]
α = 0.95
s = quantile(Chi(3), α)
X = rand(MvNormal(Σ), 1000)
scatter(X[1,:], X[2,:],X[3,:], aspect_ratio = :equal, ms = 1.2, c = 1 .+ 2*[malo_dist(x, Σ, s) for x in eachcol(X)], label = :none)
plot!(ellipsoid([0,0,0], Σ, s; segments = 150), label = "ellipsoid of 95% confidence", alpha = 0.6)

# Show eigenvectors
eg = eigen(inv(Σ))
λ = eg.values
v = eg.vectors
[plot!(tuple([[0, s*λ[i]^(-1/2)*c[j]] for j in axes(Σ,1)]...), label = :none, c=4,arrow=true, lw = 2) for (i,c) in enumerate(eachcol(v))]

xlabel!("x")
ylabel!("y")
zlabel!("z")
ylims!(-4,4)
xlims!(-4,4)
zlims!(-4,4)
```
"""
function ellipsoid(center::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}, radius=1.0; segments=60)
    # Generate angles
    n = length(center)
    @assert size(Σ, 1) == n
    @assert issymmetric(Σ)

    degvec = LinRange(0, 2π, segments)

    # Create a grid of circle/sphere coordinates
    if n == 3
        grid = collect(product(degvec, degvec))
        sphere_points = hcat(map(ecoord, grid)...)
    elseif n == 2
        sphere_points = hcat(ecoord.(degvec)...)
    end

    # Compute Cholesky decomposition with pivoting
    C = cholesky(Σ, RowMaximum())
    Q = C.U
    P = C.p
    Q = Q[:, invperm(P)]
    # Transform sphere/circle points to ellipse/ellipsoid
    vertices = hcat([center + radius * (Q' * point) for point in eachcol(sphere_points)]...)

    return tuple([x for x in eachrow(vertices)]...)
end

"""
    ecoord(angles::NTuple{2})
    ecoord(θ::Real)
Function to compute coordinates on the unit sphere/circle
"""
function ecoord(angles::NTuple{2})
    ϕ, θ = angles
    return [cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)]
end

ecoord(θ::Real) = [cos(θ), sin(θ)]

""" 
    ellipsoid_area(Σ, radius)
Compute the volume of a circle/sphere of radius `radius` definied by the set of point `x` such that `x' Σ x ≤ radius^2`
# Arguments
- `Σ`: Symmetric matrix (typically a covariance matrix).
- `radius`: Radius of the ellipsoid.  

In 2D: V = π a b = π r^2 λ₁^(-1/2) λ₂^(-1/2)
In 3D: V = 4π/3 a b c = 4π/3 r^2 λ₁^(-1/2) λ₂^(-1/2) λ₃^(-1/2)
"""
function ellipsoid_area(Σ, radius)
    n = size(Σ, 1)
    vals = eigvals(inv(Σ))
    if n == 3
        f = 4/3
    elseif n ==2
        f = 1
    end
    return radius^n*f*π*prod(vals.^(-1/2))
end

"""
    inABCEllipsoid(ABCresults, θ_test, α)
Using the ABC selected sample to estimate the posterior distribution, it computes the associated confidence ellispse, and Mahalanobis distance. It tests if this distance is bellow the `Chisq(dim)` `α`-quantile where `dim` is the dimension of the input vector.
"""
function inABCEllipsoid(ABCresults::AbstractVector, θ_test::AbstractMatrix, thr)
    N = length(ABCresults)
    @assert N == size(θ_test, 2)
    ν = size(θ_test, 1) # dimension of the parameters
    InOrNot = zeros(Bool, N)
    for (i, x) in enumerate(ABCresults)
        m = mean(x, dims = 2)[:, 1]
        C = cov(x')
        ## squared Mahalanobis distance
        @views InOrNot[i] = dot(m - θ_test[:, i], inv(C), m - θ_test[:, i]) < thr^2
    end
    return count(InOrNot) / N
end

function AreaABCEllipsoid(ABCresults::AbstractVector, thr)
    N = length(ABCresults)
    area = zeros(N)
    for (i, x) in enumerate(ABCresults)
        C = cov(x')
        ## squared Mahalanobis distance
        area[i] = ellipsoid_area(C, thr)
    end
    return area
end
## * HDR * ##
# Only 2D for now

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


#* Rectangle *#

inABCRectangle(df_results) = count(prod(hcat([df.:q_low .≤ df.:θ .≤ df.:q_high for df in df_results]...), dims=2)) / nrow(df_results[1])
AreaABCRectangle(df_results) = prod(hcat([df.:q_high - df.:q_low for df in df_results]...), dims=2)

# """
#     confidence_ellipse2D(x::AbstractVector, y::AbstractVector, s; n_points = 100)
#     confidence_ellipse2D(X::AbstractMatrix, s; n_points = 100)
# Create the points forming the `s`-confidence ellipse of `X` (or `x` and `y`).
# To have a `α`-confidence region choose `s = quantile(Chi(2), α)` 
# Can be used with `Plots.jl` and other plotting libraries.
# ```julia
# using Plots
# using Distributions
# Σ = [1 0; 0 1]
# X = rand(MvNormal(Σ), 2000)
# α = 0.95
# s = quantile(Chi(2), α)
# scatter(X[1,:], X[2,:])
# plot!(confidence_ellipse2D(X', s))
# ```
# Idea from https://github.com/CarstenSchelp/CarstenSchelp.github.io/blob/master/LICENSE
# """
# function confidence_ellipse2D(x::AbstractVector, y::AbstractVector, s; n_points = 100, cov_matrix = cov(hcat(x, y)))
#     if length(x) != length(y)
#         throw(ArgumentError("x and y must be the same size"))
#     end

    
#     pearson = cov_matrix[1, 2] / sqrt(cov_matrix[1, 1] * cov_matrix[2, 2])

#     ell_radius_x = sqrt(1 + pearson)
#     ell_radius_y = sqrt(1 - pearson)

#     scale_x = sqrt(cov_matrix[1, 1]) * s
#     mean_x = mean(x)

#     scale_y = sqrt(cov_matrix[2, 2]) * s
#     mean_y = mean(y)

#     # Generate the ellipse points
#     t = LinRange(0, 2π, n_points)
#     ellipse_x = ell_radius_x * cos.(t)
#     ellipse_y = ell_radius_y * sin.(t)

#     # Rotation and scaling transformation
#     rotation_matrix = [cos(π/4) -sin(π/4); 
#                        sin(π/4)  cos(π/4)]
#     rotated_ellipse = rotation_matrix * hcat(ellipse_x, ellipse_y)'
#     scaled_ellipse = hcat(scale_x * rotated_ellipse[1, :], scale_y * rotated_ellipse[2, :])

#     # # Translate the ellipse
#     translated_ellipse_x = scaled_ellipse[:, 1] .+ mean_x
#     translated_ellipse_y = scaled_ellipse[:, 2] .+ mean_y

#     return (translated_ellipse_x, translated_ellipse_y)
# end

# confidence_ellipse2D(X::AbstractMatrix, s; n_points = 100, cov_matrix = cov(X)) = confidence_ellipse2D(X[:,1], X[:,2], s; n_points = n_points, cov_matrix = cov_matrix)

# function areaellipse(X, s)
#     cov_matrix = cov(X)
#     return areaellipseΣ(cov_matrix, s)
# end

# function areaellipseΣ(cov_matrix, s)
#     n = size(cov_matrix, 1)
#     σ = [sqrt(cov_matrix[i, i]) for i in 1:n]
#     cor_matrix = cov2cor(cov_matrix)
#     ρ = filter(!iszero, triu(cor_matrix)-Diagonal(cor_matrix))#[cor_matrix[i,j] for j in i+1:n, j in 1:n-1]
#     return prod(sqrt(1-r^2) for r in ρ)*prod(σ)*s^2
# end

# function areaellipseΣ(cov_matrix, s)
#     ρ = cov_matrix[1, 2] / sqrt(cov_matrix[1, 1] * cov_matrix[2, 2])
#     σ1 = sqrt(cov_matrix[1, 1])
#     σ2 = sqrt(cov_matrix[2, 2])
#     return sqrt(1-ρ^2)*σ1*σ2*s^2
# end
# aa = [ABCMethods.areaellipse(results_posterior_cnn[i]',quantile(Chi(2), 0.95))
# for i in eachindex(results_posterior_cnn)]
# bb = [ABCMethods.areaellipseΣ(MC_testD[2][:, :, i],qhatD) for i in eachindex(results_posterior_cnn)]
# a = score_MultiDim(θ_test_Norm, MC_testD...) .< qhatD
# b = prod(hcat([df.:q_low .≤ df.:θ .≤ df.:q_high for df in df_cnn_result]...), dims=2)
# sortperm(bb)
# findall(b[:,1] .== 0 .&& aa.>1.5bb .&& a.==1)
# findall(aa.>1.5bb)