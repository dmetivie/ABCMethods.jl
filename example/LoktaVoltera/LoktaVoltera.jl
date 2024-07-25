using Markdown#hide
cd(@__DIR__)#hide
import Pkg#hide
Pkg.activate(".")#hide

md"""
# ABC-Conformal - Discrete Lokta Voltera example

Given a simulation models of Stocastic (discrete) Lokta Voltera[^model] we construct and compare three ABC methods:
- Standard ABC
- ABC-CNN: ABC method as described by [Åkesson et al. - 2021](https://ieeexplore.ieee.org/abstract/document/9525290).
- ABC-Conformal (ours): ABC method completly free of summary statistics and threshold selection as described in [Baragatti et al. - 2024](https://arxiv.org/abs/2406.04874).

The goal is to recover the three parameters of the model given an observation.

[^model]: see [D Prangle - 2017](https://projecteuclid.org/journals/bayesian-analysis/volume-12/issue-1/Adapting-the-ABC-Distance-Function/10.1214/16-BA1002.full) for an explanation of the model.
"""

md"""
## Packages & Settings

To install and work with the same environment i.e. package version just do the following in the same folder as the `Project.toml` (and optionally also a `Manifest.toml`)
```julia
import Pkg
Pkg.pkg"registry add https://github.com/dmetivie/LocalRegistry" # where I currently store `ABCConformal.jl` and `ConcreteDropoutLayer.jl` packages
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
## Specific to Lokta Voltera example
"""

using JumpProcesses # has Gillespie Methods

using Catalyst # Fancy model definition


md"""
## LK simulations
"""

md"""
### Model

We simulate a stocastic Lokta Voltera model during `t_span = (0, 36)`.
We assume observations at `ts = 0:2:36`.
"""

LK_model = @reaction_network begin
    α, PREY --> 2PREY
    β, PREY + PREDATOR --> 2PREDATOR
    γ, PREDATOR --> 0
end

md"""
### Simulator

To make things challenging for an inference standpoint[^loglik], we only keep simulations with strictly positive number Prey and Predator number.

We use the Gillespie algorithm (1977) and the Julia implementation with [JumpProcesses.jl](https://docs.sciml.ai/JumpProcesses/stable/) package.
Note that compared to the `R` package [GillespieSSA2](https://github.com/rcannood/GillespieSSA2) we rougthly have a `x130` speedup with Julia.
In addition it is much more stable i.e. it handles better divergin trajectories. In our `R` version we had to use (for time constrains) the approximated `ssa_etl(tau = 1e-2)` method. 
It produces a significantly different distribution of non zero trajectory (smaller variance) when compared to the `ssa_exact()` (`Direct()` in Julia) method, making the inference problem easier.


[^loglik]: In particular, any likelihood based method would be intractable. 
"""

condition(u, t, integrator) = u[2] == 0

affect!(integrator) = terminate!(integrator)

cb = DiscreteCallback(condition, affect!)

function output_func(sol, i)
    (θ=sol.prob.p, y=sol.u), false
end

function sample_from_θ(N, distθ::Distribution; saveat=2.0, tspan=(0.0, 36.0), u₀=[:PREY => 50, :PREDATOR => 100], jump_model=LK_model, more=1000, batch_size=ifelse(3N > 10^5, 10^5, 3N), cb=cb)
    θ = exp.(rand(distθ, more * N))
    p = (:α => 0.0, :β => 0.0, :γ => 0.0)
    n = Integer(tspan[end] ÷ saveat + 1)
    prob = DiscreteProblem(jump_model, u₀, tspan, p)
    jump_prob = JumpProblem(jump_model, prob, Direct(); save_positions=(false, false))

    prob_func = let p = p
        ## https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/#Example-2:-Solving-an-SDE-with-Different-Parameters
        (jump_prob, i, repeat) -> begin
            jump_prob = remake(jump_prob, p=θ[:, i])
        end
    end

    function reduction(U, batch, I)
        nonzero_scenario = filter(u -> prod(u.y[end]) != 0, batch)
        if length(nonzero_scenario) > 0
            U[:y] = cat(U[:y], [permutedims(reduce(hcat, u.y)) for u in nonzero_scenario]...; dims=3)
            U[:θ] = hcat(U[:θ], reduce(hcat, [u.θ for u in nonzero_scenario]))
        end
        finished = size(U[:y], 3) ≥ N
        U, finished
    end

    ensemble_prob = EnsembleProblem(jump_prob, prob_func=prob_func, output_func=output_func, reduction=reduction, u_init=Dict(:θ => zeros(length(distθ), 0), :y => zeros(Int, n, length(u₀), 0)))
    sol = solve(ensemble_prob, SSAStepper(), trajectories=more * N, saveat=saveat, batch_size=batch_size, callback=cb)
    return sol.u[:θ][:, 1:N], sol.u[:y][:, :, 1:N]
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

The three parameters prior is the same as in [D Prangle - 2017](https://projecteuclid.org/journals/bayesian-analysis/volume-12/issue-1/Adapting-the-ABC-Distance-Function/10.1214/16-BA1002.full).

"""

dist = product_distribution([Uniform(-6, 2), Uniform(-6, 2), Uniform(-6, 2)])

dim_θ = length(dist)

dim_y = 2

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

abc_method = ABC_Nearestneighbours(α_ABC, 𝕃2)

@time "ABC" results_posterior_abc = ABC_selection(y_test, y_train, θ_train_Norm, abc_method)

md"""
### Results
"""

df_abc_result = ABC2df(results_posterior_abc, θ_test_Norm)

let
    dfs = df_abc_result
    orders = [sortperm(df.θ) for df in dfs]

    plt_abc = [@df df plot(range(extrema([:θ; :θ_hat])..., length=10), range(extrema([:θ; :θ_hat])..., length=10), xlabel="True", ylabel="Estimated ", c=:red, s=:dot, lw=2, label=:none, aspect_ratio=true) for (i, df) in enumerate(dfs)]
    [@df df plot!(plt_abc[i], :θ[orders[i]], :q_low[orders[i]], fill=(:q_high[orders[i]], 0.4, :gray), label=:none, lw=0) for (i, df) in enumerate(dfs)]
    [@df df scatter!(plt_abc[i], :θ, :θ_hat, label=:none, c=1, alpha=0.75, left_margin=4Plots.Measures.mm) for (i, df) in enumerate(dfs)]

    plot(plt_abc..., layout=(1, 3), size=(800, 600))
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
        Dense(512 => 128, tanh),
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
    plot(plt_abc_cnn..., layout=(1, 3), size=(800, 600), left_margin=4Plots.Measures.mm)
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
        Dense(512 => 128, tanh) |> freeze,
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

    plot(plt_abc_c..., layout=(1, 3), size=(800, 600))
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