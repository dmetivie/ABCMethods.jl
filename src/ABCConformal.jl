module ABCConformal

using DataFrames, DataFramesMeta
using StatsBase

using Lux # Deep Learning package
using Optimisers
using Zygote # AD package
using Accessors # Play with nested struc
using Random

include("abc.jl")
include("abc_conformal.jl")
include("samples.jl")
include("validation_metric.jl")
include("NN.jl")

export my_norm, my_unnorm
export summary_method, 𝕃2
export ABC_Nearestneighbours, ABC_selection, ABC2df
export train_NN, compute_loss_heteroscedastic, compute_loss_heteroscedastic_w_reg, compute_loss_mse, ini_manually_CNN2CD
export MC_predict
export conformilize

end
