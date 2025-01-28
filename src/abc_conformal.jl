"""
	MC_predict_MultiDim(model_state, X::AbstractArray, n_samples=1000; dev=gpu_device(), dim_out=model_state.model[end].layers[1].out_dims, rescale = identity, rescale_var = identity)
For a `model_state` returning a value and associated `logvar`, this function computes for each slice of `X`

- the mean of `n_samples` Monte Carlo simulations where the randomness comes from the (Concrete)Dropout layers.
- The overall covariance uncertainty sum of aleatoric (Diagonal matrix of the `exp(logvar term)`) and epistemic uncertainty (classic covariance matrix). This uncertainty as the dimension of a variance.
"""
function MC_predict_MultiDim(model_state, X::AbstractArray, n_samples=1000; dev=gpu_device(), dim_out=model_state.model[end].layers[1].out_dims, rescale = identity, rescale_var = identity)
    st = model_state.states
    ps = model_state.parameters
    model = model_state.model

    dim_N = ndims(X)
    mean_arr = similar(X, dim_out, size(X, dim_N))
    var_dev_arr = similar(X, dim_out, dim_out, size(X, dim_N))

    X = X |> dev
    X_in = similar(X, size(X)[1:end-1]..., n_samples) |> dev

    for (i, x) in enumerate(eachslice(X, dims=dim_N, drop=false))
        X_in .= x

        predictions, st = model(X_in, ps, st)
        θs_MC, logvars = predictions |> cpu_device()
        θs_MC .= θs_MC |> rescale
        θ_hat = mean(θs_MC, dims=2) # predictive_mean 
        epistemic = cov(θs_MC') # multidim
        var_mean = Diagonal(mean(exp.(logvars), dims=2) |> rescale_var |> vec ) # aleatoric_uncertainty 
        total_var = epistemic + var_mean # epistemic + aleatoric uncertainty
        mean_arr[:, i] .= θ_hat
        var_dev_arr[:, :, i] .= total_var
    end

    return mean_arr, var_dev_arr
end

"""
    MC_predict(model_state, X::AbstractArray, n_samples=1000; dev=gpu_device(), dim_out=model_state.model[end].layers[1].out_dims)
For a `model_state` returning a value and associated `logvar`, this function computes for each slice of `X`

- the mean of `n_samples` Monte Carlo simulations where the randomness comes from the (Concrete)Dropout layers.
- The overall uncertainty sum of aleatoric (`exp(logvar)`) and epistemic (classic `std`) uncertainty. This uncertainty as the dimension of a standard deviation.
"""
function MC_predict(model_state, X::AbstractArray, n_samples=1000; dev=gpu_device(), dim_out=model_state.model[end].layers[1].out_dims)
    st = model_state.states
    ps = model_state.parameters
    model = model_state.model

    dim_N = ndims(X)
    mean_arr = similar(X, dim_out, size(X, dim_N))
    std_dev_arr = similar(X, dim_out, size(X, dim_N))

    X = X |> dev
    X_in = similar(X, size(X)[1:end-1]..., n_samples) |> dev

    for (i, x) in enumerate(eachslice(X, dims=dim_N, drop=false))
        X_in .= x

        predictions, st = model(X_in, ps, st)
        θs_MC, logvars = predictions |> cpu_device()

        θ_hat = mean(θs_MC, dims=2) # predictive_mean 

        # θ2_hat = mean(θs_MC .^ 2, dims=2) # unidim (only variance on diagonals)
        epistemic = [var(d) for d in eachrow(θs_MC)]
        var_mean = mean(exp.(logvars), dims=2) # aleatoric_uncertainty 
        total_var = epistemic + var_mean
        std_dev = sqrt.(total_var)

        mean_arr[:, i] .= θ_hat
        std_dev_arr[:, i] .= std_dev
    end

    return mean_arr, std_dev_arr
end

"""
	MC2df(predictive_mean, overall_uncertainty, true_θ)
Return a DataFrame with the following column `["θ", "θ_hat", "σ_tot"]`
"""
MC2df(predictive_mean, overall_uncertainty, true_θ) = [DataFrame(hcat(true_θ[i, :], predictive_mean[i, :], overall_uncertainty[i, :]), ["θ", "θ_hat", "σ_tot"]) for i in axes(true_θ, 1)]

# Conformal functions

"""
	q_hat_conformal(x_true::AbstractVector, x_hat::AbstractVector, α, σ=1)
    q_hat_conformal(x_true::AbstractMatrix, x_hat::AbstractMatrix, α, V::AbstractArray)
    q_hat_conformal(MC_cal::Tuple, θ_cal, α)
Estimate the conformal quantile of level `α`. To compute the score σ can be specified as a vecor or number. `σ` is a proxy for our confidence on the estimation.
"""
function q_hat_conformal(x_true::AbstractVector, x_hat::AbstractVector, α, σ=1)
    n = length(x_true)
    q_level = ceil((n + 1) * (1 - α)) / n
    score = abs.(x_true - x_hat) ./ σ
    return sort(score)[ceil(Int, n * q_level)]
end

"""
	conformilize!(df_test, df_cal, α)
Given two DataFrame one of test `df_test` and one of calibration `df_cal`, it estimate (and add in place) the conformal quantile low/hight for each observation.
"""
function conformilize!(df_test::DataFrame, df_cal::DataFrame, α)
    q̂ = q_hat_conformal(df_cal.:θ, df_cal.:θ_hat, α, df_cal.:σ_tot)
    @transform!(df_test,
        :q_low = :θ_hat - q̂ * :σ_tot,
        :q_high = :θ_hat + q̂ * :σ_tot
    )
end

function conformilize(df_test, df_cal, α)
    df = copy(df_test)
    conformilize!(df, df_cal, α)
    return df
end

function conformilize(MC_test::NTuple{2}, MC_cal::NTuple{2}, θ_test, θ_cal, α_conformal)
    df_cal = MC2df(MC_cal..., θ_cal)
    df_cd_test_conformal = MC2df(MC_test..., θ_test)

    conformilize!.(df_cd_test_conformal, df_cal, α_conformal)

    return df_cd_test_conformal
end

score_MultiDim(x_true, x_hat, V) = @views sqrt.([(x_true[:,i] - x_hat[:,i])' * inv(V[:,:,i]) * (x_true[:,i] - x_hat[:,i]) for i in axes(x_true, 2)])

function q_hat_conformal(x_true::AbstractMatrix, x_hat::AbstractMatrix, α, V::AbstractArray)
    n = size(x_true, 2)
    score = score_MultiDim(x_true, x_hat, V)
    q_level = ceil((n + 1) * (1 - α)) / n
    return sort(score)[ceil(Int, n * q_level)]
end

function q_hat_conformal(MC_cal::Tuple, θ_cal, α_conformal)
    xcal, Vcal = MC_cal
    return q_hat_conformal(θ_cal, xcal, α_conformal, Vcal)
end
