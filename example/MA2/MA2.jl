cd(@__DIR__)#hide
import Pkg#hide
Pkg.activate(".")#hide
using Markdown#hide

md"""
# ABC-Conformal - Moving Average (2) example

Given a simulation models of MA(2)[^model] we construct and compare three ABC methods:
- Standard ABC
- ABC-CNN: ABC method as described by [Åkesson et al. - 2021](https://ieeexplore.ieee.org/abstract/document/9525290).
- ABC-Conformal (ours): ABC method completly free of summary statistics and threshold selection as described in [Baragatti et al. - 2024](https://arxiv.org/abs/2406.04874).

The goal is to recover the three parameters of the model given an observation.

[^model]: This is a famous toy example in Bayesian inferance, see 
"""

md"""
## Packages & Settings

To install and work with the same environment i.e. package version just do the following in the same folder as the `Project.toml` (and optionally also a `Manifest.toml`)
```julia
import Pkg
pkg"registry add https://github.com/dmetivie/LocalRegistry" # where I currently store `ABCConformal.jl` and `ConcreteDropoutLayer.jl` packages
Pkg.activate(".")
Pkg.instantiate()
```
"""

using ABCConformal

md"""
Some generic packages
"""

using DataFrames, DataFramesMeta

using Random

using Distributions

using StatsPlots, LaTeXStrings

default(fontfamily="Computer Modern")

md"""
Deep Learning packages
"""

using MLUtils
using Lux
using LuxCUDA # comment if you use CPU
dev = gpu_device() # cpu_device()
using Lux: freeze
## using Optimisers, Zygote # only if you need more control on Optimiszer and AD
## using Accessors # to change nested structures. Useful to play with Lux.TrainState
using ConcreteDropoutLayer 

md"""
## Specific to MA(2) example
"""

using ToeplitzMatrices

md"""
## LK simulations
"""

md"""
### Model

We create a new distribution
"""

struct TriangularUniform{T<:Real} <: ContinuousMultivariateDistribution
    x_min::T
    x_max::T
    y_min::T
    y_max::T
end

function inTriangle(x, x_min, x_max, y_min, y_max)
    x₁ = x_min < x[1] < x_max
    x₂ = y_min < x[2] < y_max
    x₊ = y_min < x[1] + x[2] && x[1] - x[2] < y_max
    return x₁ && x₂ && x₊
end

function Distributions.insupport(d::TriangularUniform, x::AbstractVector)
    inTriangle(x, d.x_min, d.x_max, d.y_min, d.y_max)
end

md"""
If one wanted to use a likelihood based method using for example the great [Turing.jl](https://turinglang.org/) package, one could define the `logpdf` of this distribution as
```julia
function Distributions.logpdf(d::TriangularUniform, x::AbstractVector)
    if !insupport(d, x)
        return Distributions.xlogy(one(eltype(d.y_max)), zero(eltype(x))) # -Inf
    end
    return log((d.x_max - d.x_min) * (d.y_max - d.y_min) / 2) # log(1/area of triangle)
end
```

Here to illustrate our ABC method we won't be using it. We will need to only have a sampling function.
"""

Distributions.length(d::TriangularUniform) = 2

function Distributions.rand(rng::AbstractRNG, d::TriangularUniform)
    cond = false
    ## I need to add that because of global scope issue
    θ₁ = rand(rng, Uniform(d.x_min, d.x_max))
    θ₂ = rand(rng, Uniform(d.y_min, d.y_max))
    cond = θ₁ + θ₂ > -1 && θ₁ - θ₂ < 1
    while cond == false
        θ₁ = rand(rng, Uniform(d.x_min, d.x_max))
        θ₂ = rand(rng, Uniform(d.y_min, d.y_max))
        cond = θ₁ + θ₂ > d.y_min && θ₁ - θ₂ < d.y_max
    end
    return [θ₁, θ₂]
end

md"""
### Simulator


"""

function SquaredTridiagonal(θ::AbstractArray{F}, T) where {F<:Real}
    return SymmetricToeplitz([1 + sum(abs2, θ), θ[1] * (1 + θ[2]), θ[2], zeros(F, T - 3)...])
end

function sample_from_θ(N, dist; n = 100)
    θ = zeros(length(dist), N)
    y_sampled = zeros(n, 1, N) # format to use with Neural Networks
    for i in 1:N
        θ[:, i] .= rand(dist)
        M = SquaredTridiagonal(θ[:, i], n)
        y_sampled[:, 1, i] = rand(MvNormal(M))
    end
    return θ, y_sampled
end
md"""
# Simulation of the parameters and of the datasets
"""

Random.seed!(MersenneTwister(0))

md"""
## Samples
"""

md"""
### Create training/test/validation/calibration sets


"""

dist = TriangularUniform(-2, 2, -1, 1)

dim_θ = length(dist)

dim_y = 1

nbsimus_test = 10^3

nbsimus_train = 10^5

nbsimus_cal = 10^3

@time "Sample test" θ_test, y_test = sample_from_θ(nbsimus_test, dist) # test set from prior

@time "Sample validation" θ_val, y_val = sample_from_θ(nbsimus_test, dist) # validation set

@time "Sample calibration" θ_cal, y_cal = sample_from_θ(nbsimus_cal, dist) # calibration set

@time "Sample train" θ_train, y_train = sample_from_θ(nbsimus_train, dist) # training set from prior

md"""
### Normalization
"""

θ_train_Norm, m_θ_train, s_θ_train = my_norm(θ_train; ms=true)

θ_test_Norm, m_θ_test, s_θ_test = my_norm(θ_test; ms=true)

θ_val_Norm, m_θ_val, s_θ_val = my_norm(θ_val; ms=true)

θ_cal_Norm, m_θ_cal, s_θ_cal = my_norm(θ_cal; ms=true)

md"""
### Batch

Most deep learning framework (Lux.jl, Pytorch, TensorFlow) works with Floats. In particular `Float32` are less memory consuming.
"""

batch_size = 128
data_train = DataLoader((y_train .|> Float32, θ_train_Norm .|> Float32), batchsize=batch_size)
data_val = (y_val .|> Float32, θ_val_Norm .|> Float32)

md"""
## Standard ABC
"""

α_ABC = 0.001 # % of sample kept

md"""
Summary statistics for the MA(2) model.
"""
τ₁(x) = sum(x[j] * x[j-1] for j in 2:length(x))
τ₂(x) = sum(x[j] * x[j-2] for j in 3:length(x))
abc_method = ABC_Nearestneighbours(α_ABC, x -> [τ₁(x), τ₂(x)], 𝕃2)

@time "ABC" results_posterior_abc = ABC_selection(y_test, y_train, θ_train_Norm, abc_method, all_samples = false)

md"""
### Results
"""

df_abc_result = ABC2df(results_posterior_abc, θ_test_Norm)

let
    dfs = df_abc_result
    orders = [sortperm(df.θ) for df in dfs]

    plt_abc = [@df df plot(range(extrema([:θ; :θ_hat])..., length=10), range(extrema([:θ; :θ_hat])..., length=10), xlabel="True", ylabel="Estimated ", c=:red, s=:dot, lw=2, label=:none, aspect_ratio=true) for (i, df) in enumerate(dfs)]
    [@df df plot!(plt_abc[i], :θ[orders[i]], :q_low[orders[i]], fill=(:q_high[orders[i]], 0.4, :gray), label=:none, lw=0) for (i, df) in enumerate(dfs)]
    [@df df scatter!(plt_abc[i], :θ, :θ_hat, label=L"c_%$i", c=1, alpha=0.75, left_margin=4Plots.Measures.mm) for (i, df) in enumerate(dfs)]

    plot(plt_abc..., layout=(1, 2), size=(800, 600))
end

md"""
Summary metrics for the method
"""
summary_abc = summary_method.(df_abc_result)

md"""
## CNN as summary statistics for ABC (Akesson et al.)

Same training set. We use a validation set to select the best model version during training.
"""

md"""
### Training
"""
function build_modelCNN(dim_input, dim_out)
    model = Lux.Chain(
        Conv((2,), dim_input => 128, tanh),
        MaxPool((2,)),
        Conv((2,), 128 => 128, tanh),
        MaxPool((2,)),
        FlattenLayer(),
        Dense(3072 => 128, tanh),
        Dense(128 => 128, tanh),
        Dense(128 => dim_out)
    )
    return model
end

epochs_CNN = 100

model_CNN = build_modelCNN(dim_y, dim_θ)
@time "Training ABC CNN" model_state_out_CNN, loss_train, loss_val = train_NN(model_CNN, epochs_CNN, data_train, data_val, compute_loss_mse; dev=dev)

md"""
Plot training loss
"""
let
    p_train = plot(loss_train, label="CNN train", title="Train loss")
    xlabel!("Epoch")
    ylabel!("loss")
    p_test = plot(loss_val, label="CNN val", title="Validation loss")
    xlabel!("Epoch")
    ylabel!("loss")
    plot(p_train, p_test)
end

md"""
### ABC after CNN

The trainied model is used a the summary statistic function. Here we have to be careful with computation on GPU/CPU
"""

η_CNN(x) = model_state_out_CNN.model(x |> dev, model_state_out_CNN.parameters, model_state_out_CNN.states) |> first |> cpu_device()

abc_method_cnn = ABC_Nearestneighbours(α_ABC, η_CNN, 𝕃2)

@time "ABC selection with η_CNN" results_posterior_cnn = ABC_selection(y_test .|> Float32, y_train .|> Float32, θ_train_Norm, abc_method_cnn)

md"""
### Results
"""

df_cnn_result = ABC2df(results_posterior_cnn, θ_test_Norm)

let
    dfs = df_cnn_result
    orders = [sortperm(df.θ) for df in dfs]

    plt_abc_cnn = [@df df plot(range(extrema([:θ; :θ_hat])..., length=10), range(extrema([:θ; :θ_hat])..., length=10), xlabel="True", ylabel="Estimated ", c=:red, s=:dot, lw=2, label=:none, aspect_ratio=true) for (i, df) in enumerate(dfs)]
    [@df df plot!(plt_abc_cnn[i], :θ[orders[i]], :q_low[orders[i]], fill=(:q_high[orders[i]], 0.4, :gray), label=:none, lw=0) for (i, df) in enumerate(dfs)]
    [@df df scatter!(plt_abc_cnn[i], :θ, :θ_hat, label=L"c_%$i", c=1, alpha=0.75) for (i, df) in enumerate(dfs)]
    plot(plt_abc_cnn..., layout=(1, 2), size=(800, 600), left_margin=4Plots.Measures.mm)
end

md"""
Summary metrics for the method
"""
summary_abc_cnn = summary_method.(df_cnn_result)

md"""
## ABC-Conformal
"""

md"""
### Model with Concrete Dropout

The model is harder to train than the previous one. Using the previous model parameters one can freeze in place its parameter to only see the effect of Concrete Dropout and the heteroscedastic loss.
In this example, we don't add here regularization term for Dropout term in the loss (the parameter `λp` is a bit hard to tune).
"""

md"""
### Training
"""
function build_modelCNN_CD_freeze(dim_input, dim_output)
    Lux.Chain(
        Conv((2,), dim_input => 128, tanh) |> freeze, # Conv1D
        ConcreteDropout(; dims=(2, 3)), # ConcreteDropout1D
        MaxPool((2,)),
        Conv((2,), 128 => 128, tanh) |> freeze,
        ConcreteDropout(; dims=(2, 3)),
        MaxPool((2,)),
        FlattenLayer(),
        Dense(3072 => 128, tanh) |> freeze,
        ConcreteDropout(),
        Dense(128 => 128, tanh) |> freeze,
        ConcreteDropout(),
        Parallel(nothing,
            Lux.Chain(Dense(128 => dim_output) |> freeze, ConcreteDropout()), # mean
            Lux.Chain(Dense(128 => dim_output), ConcreteDropout())  # logvar
        )
    )
end

model_CNN_CD_freeze = build_modelCNN_CD_freeze(dim_y, dim_θ)
model_state_out_CD_ini = ini_manually_CNN2CD(model_CNN_CD_freeze, model_state_out_CNN)

epochs_CD = 50
@time "Training ABC ConcreteDropout freeze" model_state_out_CD, loss_train_CD, loss_val_CD = train_NN(model_state_out_CD_ini, epochs_CD, data_train, data_val, compute_loss_heteroscedastic; dev=gpu_device())

md"""
Compute MSE loss only (not heteroscedastic loss)
"""
println("MSE CNN only ", MSELoss()(first(model_state_out_CNN.model(y_val .|> Float32 |> dev, model_state_out_CNN.parameters, model_state_out_CNN.states) |> cpu_device()), θ_val_Norm .|> Float32))
println("MSE with Concrete Dropout and heterocedastic loss ", MSELoss()(first(model_state_out_CD.model(y_val .|> Float32 |> dev, model_state_out_CD.parameters, model_state_out_CD.states) |> first |> cpu_device()), θ_val_Norm .|> Float32))

md"""
Print all learned Dropout rates.
"""
path_cd = regularization_infos(model_state_out_CD, true)
println.(path_cd, " ", get_regularization(model_state_out_CD.parameters, path_cd));

md"""
### MC Predict
"""

@time MC_test = MC_predict(model_state_out_CD, y_test .|> Float32, 1000; dim_out=dim_θ)

md"""
Same for the calibration set
"""
@time MC_cal = MC_predict(model_state_out_CD, y_cal .|> Float32, 1000, dim_out=dim_θ)

md"""
### Conformal
As an heuristic uncertainty for the conformal procedure, we use the predictive variance, which is the sum of the aleatoric and the epistemic variances.
"""

α_conformal = 0.05
df_cd_test_conformal = conformilize(MC_test, MC_cal, θ_test_Norm, θ_cal_Norm, α_conformal)

md"""
### Results 
"""
let
    dfs = df_cd_test_conformal
    orders = [sortperm(df.θ) for df in dfs]

    plt_abc_c = [@df df plot(range(extrema([:θ; :θ_hat])..., length=10), range(extrema([:θ; :θ_hat])..., length=10), xlabel="True", ylabel="Estimated ", c=:red, s=:dot, lw=2, label=:none, aspect_ratio=true) for (i, df) in enumerate(dfs)]
    [@df df plot!(plt_abc_c[i], :θ[orders[i]], :q_low[orders[i]], fill=(:q_high[orders[i]], 0.4, :gray), label=:none, lw=0) for (i, df) in enumerate(dfs)]
    [@df df scatter!(plt_abc_c[i], :θ, :θ_hat, label=L"c_%$i", c=1, alpha=0.75) for (i, df) in enumerate(dfs)]

    plot(plt_abc_c..., layout=(1, 2), size=(800, 600))
end

md"""
Summary metrics for the method
"""
summary_abc_conformal = summary_method.(df_cd_test_conformal)


md"""
# Comparison of all methods
"""
using PrettyTables
df_results = map(1:dim_θ) do i
    vcat(summary_abc[i], summary_abc_cnn[i], summary_abc_conformal[i])
end

hl_min(col) = HtmlHighlighter(
    (data, i, j) -> (j == col) && data[i, col] == minimum(data[:, col]),
    ## MarkdownDecoration(bold=true)
    HtmlDecoration(color = "blue", font_weight = "bold")
    ## crayon"blue bold"
)

hl_method = HtmlHighlighter(
           (data, i, j) -> (j == 1),
           HtmlDecoration(color = "red")
)
for i in 1:dim_θ
    pretty_table(
        hcat(["ABC", "ABC CNN", "ABC Conformal"], Matrix(df_results[i]));
        ##    backend = Val(:markdown),
        backend = Val(:html),
        standalone = true,
        highlighters=(hl_method, hl_min(2), hl_min(3), hl_min(4), hl_min(5)),
        ## border_crayon=crayon"yellow",
        formatters=ft_printf("%5.2f", 2:4),
        ##    tf            = tf_unicode_rounded,           
        header=vcat("c$i  Method", names(df_results[i]))
    )
end

md"""
Julia settings
"""

using InteractiveUtils
InteractiveUtils.versioninfo()

if @isdefined(LuxDeviceUtils)
    if @isdefined(CUDA) && LuxDeviceUtils.functional(LuxCUDADevice)
        println()
        CUDA.versioninfo()
    end
end
