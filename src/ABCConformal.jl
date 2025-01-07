module ABCConformal

using DataFrames, DataFramesMeta
using StatsBase

using Lux # Deep Learning package
using Lux.Training: TrainState
using Optimisers
using Random
using LinearAlgebra
using Base.Iterators: product
using HighestDensityRegions, KernelDensity

include("abc.jl")
include("abc_conformal.jl")
include("samples.jl")
include("validation_metric.jl")
include("NN.jl")
include("confidence_regions.jl")

export my_norm, my_unnorm
export summary_method, 𝕃2
export ABC_NearestNeighbours, ABC_selection, ABC2df
export train_NN
export compute_loss_heteroscedastic, compute_loss_mse, heteroscedastic_loss
export MC_predict, MC_predict_MultiDim, q_hat_conformal 
export inHDR, inABCEllipsoid, ABCEllipsoidArea, inABCRectangle, ABCRectangleArea
export score_MultiDim
export ellipsoid_area, ellipsoid
export conformilize

end
