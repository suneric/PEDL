# Physics-Enhanced Deep Learning (PEDL) Project - Part One

Use simulation data to train a prediction of 2-DOF-cart-pendulum dynamic system.

## Three types of models
1. MLP-based DNN model
2. LSTM-based DNN model
3. PINN model

## Customized Loss function
Weighted data loss and physics loss in the final loss function, the weight is controlled by the params_control.alpha

## Play
1. Specify parameters in params_system, params_control, and params_training
2. Run an_experiment.m
3. View the training process and the final prediction plots