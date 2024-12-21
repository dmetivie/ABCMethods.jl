module ABCConformal

using DataFrames, DataFramesMeta
using StatsBase

using Lux # Deep Learning package
using Optimisers
using Zygote # AD package
using Accessors # Play with nested struc
using Random
using LinearAlgebra: Diagonal, dot
using HighestDensityRegions, KernelDensity

include("abc.jl")
include("abc_conformal.jl")
include("samples.jl")
include("validation_metric.jl")
include("NN.jl")

export my_norm, my_unnorm
export summary_method, 𝕃2
export ABC_Nearestneighbours, ABC_selection, ABC2df
export train_NN
export compute_loss_heteroscedastic, compute_loss_mse, heteroscedastic_loss
export MC_predict, MC_predict_MultiDim, q_hat_conformal 
export inHDR, inABCellipse, confidence_ellipse2D, score_MultiDim
export conformilize

end
