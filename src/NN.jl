# Training 

"""
	train(model, epochs, dataset, dataset_val, compute_loss; learning_rate=0.001f0, dev = cpu_device())
Train the `model` and comute at each epoch the training and testing loss
"""
function train_NN(model::Lux.Chain, epochs, dataset, dataset_val, compute_loss; learning_rate=0.001f0, dev = cpu_device())
    ## Set up models
    rng = Xoshiro(0)

    train_state = Lux.Experimental.TrainState(rng, model, Adam(learning_rate); transform_variables=dev)

    return train_NN(train_state, epochs, dataset, dataset_val, compute_loss; dev = dev)
end

"""
	train(model, epochs, dataset, dataset_val, compute_loss; learning_rate=0.001f0, dev = cpu_device())
Train the `model` and comute at each epoch the training and testing loss
"""
function train_NN(train_state::Lux.Experimental.TrainState, epochs, dataset, dataset_val, compute_loss; dev = cpu_device())
    ps = train_state.parameters
    st = train_state.states
    model = train_state.model

    ## Validation Loss
    losses_train = Float32[]
    x_val, y_val = dataset_val |> dev
    losses_val = Float32[first(compute_loss(model, ps, st, (x_val, y_val)))]
    loss = rand(Float32) # just to define loss in outer loop scope # probably better ways to do that
    best_test_state = train_state

    ## Training loop
    for epoch in 1:epochs
        issave = false
        for xy in dataset
            xy = xy |> dev
            loss, train_state = train_step(train_state, xy, compute_loss)
        end
        ps = train_state.parameters
        st = train_state.states
        loss_val = first(compute_loss(model, ps, st, (x_val, y_val)))
        if loss_val < minimum(losses_val)
            best_test_state = train_state
            issave = true
        end
        append!(losses_train, loss)
        append!(losses_val, loss_val)
        @info "Epoch $epoch train_loss = $(round(loss, digits = 4)) validation_loss = $(round(loss_val, digits = 4)) $(issave ? "Best model so far" : " ")"
    end
    return best_test_state, losses_train, losses_val
end

function train_step(train_state, xy, compute_loss)
    ## Calculate the gradient of the objective
    ## with respect to the parameters within the model:
    x, y = xy
    
    gs, loss, _, train_state = Lux.Experimental.compute_gradients(
                AutoZygote(), compute_loss, (x, y), train_state
    )
    train_state = Lux.Experimental.apply_gradients(train_state, gs)

    return loss, train_state
end

# Loss

function heteroscedastic_loss(y_pred, y_true)
    μ, log_var = y_pred
    precision = exp.(-log_var)
    return mean(precision .* (y_true - μ) .^ 2 + log_var)
end

function compute_loss_heteroscedastic(model, ps, st, (x, y))
    ŷ, st = model(x, ps, st)
    return heteroscedastic_loss(ŷ, y), st, ()
end

mse = MSELoss()

function compute_loss_mse(model, ps, st, (x, y))
    # Generate the model predictions.
    ŷ, st = model((x), ps, st)
    return mse(ŷ, y), st, ()
end

"""
Loss with regularization
Version with the added regularization suggested in the original paper. `(names_CD, names_W, input_features), λp, λW` are provided and constant during the training.
"""
function compute_loss_heteroscedastic_w_reg(model, ps, st, (x, y), (names_CD, names_W, input_features)::NTuple{3}, λp, λW)
    ŷ, st = model(x, ps, st)
    drop_rates, W = get_regularization(ps, names_CD, names_W)

    return heteroscedastic_loss(ŷ, y) + computeCD_reg(drop_rates, W, input_features, λp, λW), st, ()
end

function compute_loss_heteroscedastic_w_reg(model, ps, st, (x, y), (names_CD, input_features)::NTuple{2}, λp)
    ŷ, st = model(x, ps, st)
    drop_rates = get_regularization(ps, names_CD)

    return heteroscedastic_loss(ŷ, y) + computeCD_reg(drop_rates, input_features, λp), st, ()
end


# Conversion 
# TODO: Dirty
"""
    Create a `train_state` where all weigths common to the `model_state_out_CNN` are initialized and frozen.
    **Warning** This function has to be change depending on the design of the models.
"""
function ini_manually_CNN2CD(model_CNN_CD, model_state_out_CNN)
    rng = Xoshiro(0)
    dev =  !isa(model_state_out_CNN.parameters.layer_1.weight, Array) ? gpu_device() : cpu_device()
    optimizer = model_state_out_CNN.optimizer
    train_state = Lux.Experimental.TrainState(rng, model_CNN_CD, optimizer; transform_variables=dev)

    @reset train_state.states.layer_1.frozen_params.weight = model_state_out_CNN.parameters.layer_1.weight
    @reset train_state.states.layer_1.frozen_params.bias = model_state_out_CNN.parameters.layer_1.bias
    @reset train_state.states.layer_4.frozen_params.weight = model_state_out_CNN.parameters.layer_3.weight
    @reset train_state.states.layer_4.frozen_params.bias = model_state_out_CNN.parameters.layer_3.bias
    @reset train_state.states.layer_8.frozen_params.weight = model_state_out_CNN.parameters.layer_6.weight
    @reset train_state.states.layer_8.frozen_params.bias = model_state_out_CNN.parameters.layer_6.bias
    @reset train_state.states.layer_10.frozen_params.weight = model_state_out_CNN.parameters.layer_7.weight
    @reset train_state.states.layer_10.frozen_params.bias = model_state_out_CNN.parameters.layer_7.bias
    @reset train_state.states.layer_12.layer_1.layer_1.frozen_params.weight = model_state_out_CNN.parameters.layer_8.weight
    @reset train_state.states.layer_12.layer_1.layer_1.frozen_params.bias = model_state_out_CNN.parameters.layer_8.bias
    return train_state
end