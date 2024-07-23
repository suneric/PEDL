%% 
close all;
clear; 
clc;

sysParams = params_system();
ctrlParams = params_control();
global trainParams;
trainParams = params_training();

%% plot system motion with a sample
f1Max = 20;
tSpan = [0,5];
plot_system(sysParams, ctrlParams, f1Max, tSpan);

%% generate samples
if ~exist("\data\", 'dir')
   mkdir("data");
end
dataFile = generate_samples(sysParams, ctrlParams, trainParams);

%% train model
if ~exist("\model\", 'dir')
   mkdir("model");
end
switch trainParams.type
    case "dnn"
        modelFile = train_dnn_model(dataFile, trainParams);
    case "lstm"
        modelFile = train_lstm_model(dataFile, trainParams);
    case "pinn"
        modelFile = train_pinn_model(dataFile, trainParams);
    otherwise
        disp("unspecified type of model.")
end

%% plot prediction
f1Max = 28;
predIntervel = 10;
tSpan = [0,10];
plot_prediction(modelFile, sysParams, ctrlParams, trainParams, f1Max, predIntervel, tSpan);

%% model evaluation
avgRMSE = evaluate_model(modelFile, sysParams, ctrlParams, trainParams);
disp(["average rmse", avgRMSE])