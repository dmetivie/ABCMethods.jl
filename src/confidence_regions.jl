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
and so on.
"""
function ellipsoid_area(Σ, radius)
    n = size(Σ, 1)
    vals = eigvals(inv(Σ))
    return radius^n*unit_sphere_volume(n)*prod(vals.^(-1/2))
end

function unit_sphere_volume(n)
    if n == 0
        return 1  # Volume of a 0-dimensional sphere is 1 (a point)
    elseif n == 1
        return 2  # Volume of a 1-dimensional sphere is 2 (a line segment of length 2)
    else
        return 2π / n * unit_sphere_volume(n - 2)
    end
end

"""
    inABCEllipsoid(ABCresults, θ_test, α)
Using the ABC selected sample to estimate the posterior distribution, it computes the associated confidence ellispse, and Mahalanobis distance. It tests if this distance is bellow the `Chisq(dim)` `α`-quantile where `dim` is the dimension of the input vector.
"""
function inABCEllipsoid(ABCresults::AbstractVector, θ_test::AbstractMatrix, thr)
    N = length(ABCresults)
    return count(inABCEllipsoidList(ABCresults, θ_test, thr)) / N
end

function inABCEllipsoidList(ABCresults::AbstractVector, θ_test::AbstractMatrix, thr)
    N = length(ABCresults)
    @assert N == size(θ_test, 2)
    InOrNot = zeros(Bool, N)
    for (i, x) in enumerate(ABCresults)
        m = mean(x, dims = 2)[:, 1]
        C = cov(x')
        ## squared Mahalanobis distance
        @views InOrNot[i] = dot(m - θ_test[:, i], inv(C), m - θ_test[:, i]) < thr^2
    end
    return InOrNot
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
    return count(inHDRList(ABCresults::AbstractVector, θ_test, α))/length(ABCresults)
end

function inHDRList(ABCresults::AbstractVector, θ_test, α)
    @assert length(ABCresults) == size(θ_test, 2)
    if size(θ_test, 1) == 2
        iks_thrs = compute_thresholds2D.(ABCresults, α)
        return [pdf(iks_thrs[i][1], c[1], c[2]) > iks_thrs[i][2] for (i,c) in enumerate(eachcol(θ_test))]
    else
        @warn "kde not yet implemented for $(size(result, 1))-Dimension"
    end
end


#* Rectangle *#

inABCRectangle(df_results) = count(inABCRectangleList(df_results)) / nrow(df_results[1])
inABCRectangleList(df_results) = prod(hcat([df.:q_low .≤ df.:θ .≤ df.:q_high for df in df_results]...), dims=2)

AreaABCRectangle(df_results) = prod(hcat([df.:q_high - df.:q_low for df in df_results]...), dims=2)

# aa = [ABCMethods.ellipsoid_area(results_posterior_cnn[i]',quantile(Chi(2), 0.95))
# for i in eachindex(results_posterior_cnn)]
# bb = [ABCMethods.ellipsoid_area(MC_testD[2][:, :, i],qhatD) for i in eachindex(results_posterior_cnn)]
# a = score_MultiDim(θ_test_Norm, MC_testD...) .< qhatD
# b = prod(hcat([df.:q_low .≤ df.:θ .≤ df.:q_high for df in df_cnn_result]...), dims=2)
# sortperm(bb)
# findall(b[:,1] .== 0 .&& aa.>1.5bb .&& a.==1)
# findall(aa.>1.5bb)