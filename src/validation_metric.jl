function ABC2df(results, θ_test; q_min = 0.025, q_max = 0.975)
	θs_hat = reduce(hcat, mean.(results, dims=2))
	qs_min = reduce(hcat, [[quantile(r, q_min) for r in eachrow(S)] for S in results])
	qs_max = reduce(hcat, [[quantile(r, q_max) for r in eachrow(S)] for S in results])
	return [DataFrame(hcat(θ_test[i,:], θs_hat[i,:], qs_min[i,:], qs_max[i,:]), ["θ", "θ_hat", "q_low", "q_high"]) for i in axes(θ_test, 1)]
end

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
