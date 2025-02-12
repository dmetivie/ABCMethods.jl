# Training 

"""
	train_NN([rng], model, epochs, dataset, dataset_val, compute_loss; opt=Adam(0.001f0), dev = cpu_device(), val_loss = compute_loss, adtype)
	train_NN(train_state, epochs, dataset, dataset_val, compute_loss; dev = cpu_device(), adtype)
Train the `model` or `train_state` and comute at each epoch the training with `compute_loss` and validation loss with `val_loss`.
"""
function train_NN(train_state::TrainState, epochs, dataset, dataset_val, compute_loss; dev = cpu_device(), val_loss = compute_loss, adtype)
    ps = train_state.parameters
    st = train_state.states
    model = train_state.model

    ## Validation Loss
    x_val, y_val = dataset_val |> dev
    floattype = eltype(x_val)
    losses_train = floattype[]
    losses_val = floattype[first(val_loss(model, ps, st, (x_val, y_val)))]
    # mse_loss_val = MSELoss()(first(first(model(x_val, ps, st))), y_val)

    loss = one(floattype) # just to define loss in outer loop scope # probably better ways to do that
    best_test_state = train_state
    # @info "Epoch 0                     validation_loss = $(round(losses_val[1], digits = 4)) mse_val_loss = $(round(mse_loss_val, digits = 4)) Best model so far"
    @info "Epoch 0                     validation_loss = $(round(losses_val[1], digits = 4)) Best model so far"
    ## Training loop
    for epoch in 1:epochs
        issave = false
        for xy in dataset
            xy = xy |> dev
            loss, train_state = train_step(train_state, xy, compute_loss; adtype = adtype)
        end
        ps = train_state.parameters
        st = train_state.states
        loss_val = first(val_loss(model, ps, st, (x_val, y_val)))
        # mse_loss_val = MSELoss()(first(first(model(x_val, ps, st))), y_val)
        if loss_val < minimum(losses_val)
            best_test_state = train_state
            issave = true
        end
        append!(losses_train, loss)
        append!(losses_val, loss_val)
        # @info "Epoch $epoch train_loss = $(round(loss, digits = 4)) validation_loss = $(round(loss_val, digits = 4)) mse_val_loss = $(round(mse_loss_val, digits = 4)) $(issave ? "Best model so far" : " ")"
        @info "Epoch $epoch train_loss = $(round(loss, digits = 4)) validation_loss = $(round(loss_val, digits = 4)) $(issave ? "Best model so far" : " ")"
    end
    return best_test_state, losses_train, losses_val
end

function train_NN(rng::AbstractRNG, model, epochs, dataset, dataset_val, compute_loss; opt=Adam(0.001f0), dev = cpu_device(), val_loss = compute_loss, adtype)
    ## Set up models
    ps, st = Lux.setup(rng, model) |> dev    

    train_state = TrainState(model, ps, st, opt)
    return train_NN(train_state, epochs, dataset, dataset_val, compute_loss; dev = dev, adtype = adtype, val_loss = val_loss)
end

train_NN(model, epochs, dataset, dataset_val, compute_loss; opt=Adam(0.001f0), dev = cpu_device(), val_loss = compute_loss, adtype) = train_NN(Random.Xoshiro(0), model, epochs, dataset, dataset_val, compute_loss; opt=opt, dev = dev, val_loss = val_loss, adtype = adtype)

function train_step(train_state, xy, compute_loss; adtype)
    ## Calculate the gradient of the objective
    ## with respect to the parameters within the model:
    x, y = xy
    
    # gs, loss, _, train_state = Lux.Training.compute_gradients(
    #             AutoZygote(), compute_loss, (x, y), train_state
    # )
    # train_state = Lux.Training.apply_gradients(train_state, gs)
    # New Lux v1 API
    gs, loss, stats, train_state = Training.single_train_step!(adtype, compute_loss, (x, y), train_state)
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