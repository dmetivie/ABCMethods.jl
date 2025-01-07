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