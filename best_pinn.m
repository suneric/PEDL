%% 
close all;
clear; 
clc;

sysParams = params_system();
ctrlParams = params_control();
global trainParams;
trainParams = params_training();
trainParams.alpha = 0.7;
trainParams.dropoutFactor = 0.1;
trainParams.numLayers = 8;
trainParams.numNeurons = 256;
trainParams.learningRate = 0.001;
trainParams.numSamples = 200;
trainParams.miniBatchSize = 200;
trainParams.type = "pinn";
trainParams.numEpochs = 50;

%% generate samples
if ~exist("\data\", 'dir')
   mkdir("data");
end
dataFile = generate_samples(sysParams, ctrlParams, trainParams);

%% train model
if ~exist("\model\", 'dir')
   mkdir("model");
end

modelFile = train_pinn_model(dataFile, trainParams);

%% model evaluation
avgRMSE = evaluate_model(modelFile, sysParams, ctrlParams, trainParams);
disp(["average rmse", avgRMSE])

%% plot prediction
fRange = 8;
predIntervel = 10;
tSpan = [0,10];
plot_prediction(modelFile, sysParams, ctrlParams, trainParams, fRange, predIntervel, tSpan);
