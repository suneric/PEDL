%% 
close all;
clear; 
clc;
sysParams = params_system();
ctrlParams = params_control();
global trainParams;
trainParams = params_training();
trainParams.numSamples = 500;
trainParams.type = "lstm6";
trainParams.numEpochs = 50;
trainParams.initLearningRate = 1e-3;
trainParams.stopLearningRate = 1e-5;
trainParams.miniBatchSize = 2000;
trainParams.lrDropEpoch = 1;

%% plot system motion with a sample
% f1Max = 15;
% tSpan = [0,5];
% len = plot_system(sysParams, ctrlParams, f1Max, tSpan);
% disp(num2str(len)+" time steps");

%% generate samples
if ~exist("\data\", 'dir')
   mkdir("data");
end
f1Max = [15,35];
tSpan = [0,5];
[dataFile, fMaxRange] = generate_samples(sysParams, ctrlParams, trainParams, f1Max, tSpan);
% plot(sort(fMaxRange));
% histogram(sort(fMaxRange),trainParams.numSamples)

%% train model
if ~exist("\model\", 'dir')
   mkdir("model");
end
switch trainParams.type
    case "dnn2"
        [modelFile, trainLoss] = train_dnn_model_2(dataFile, trainParams);
    case "lstm2"
        [modelFile, trainLoss] = train_lstm_model_2(dataFile, trainParams);
    case "pinn2"
        [modelFile, trainLoss] = train_pinn_model_2(dataFile, trainParams);
    case "dnn4"
        [modelFile, trainLoss] = train_dnn_model_4(dataFile, trainParams);
    case "lstm4"
        [modelFile, trainLoss] = train_lstm_model_4(dataFile, trainParams);
    case "pinn4"
        [modelFile, trainLoss] = train_pinn_model_4(dataFile, trainParams);
    case "dnn6"
        [modelFile, trainLoss] = train_dnn_model_6(dataFile, trainParams);
    case "lstm6"
        [modelFile, trainLoss] = train_lstm_model_6(dataFile, trainParams);
    case "pinn6"
        [modelFile, trainLoss] = train_pinn_model_6(dataFile, trainParams);
    % case "pirn2"
    %     [modelFile, trainLoss] = train_pirn_model_2(dataFile, trainParams);
    % case "pirn4"
    %     [modelFile, trainLoss] = train_pirn_model_4(dataFile, trainParams);
    % case "pirn6"
    %     [modelFile, trainLoss] = train_pirn_model_6(dataFile, trainParams);
    otherwise
        disp("unspecified type of model.")
end

%% plot training curve
% figure('Position',[500,100,800,400]); 
% tiledlayout("vertical","TileSpacing","tight")
% x = 1:length(trainLoss);
% y = trainLoss(x);
% smoothed_y = smoothdata(y,'gaussian');
% plot(x,y,'b-',x,smoothed_y,'r-',"LineWidth",2);
% xlabel("Iteration","FontName","Arial")
% ylabel("Loss","FontName","Arial")
% legend("Original","Smoothed","location","best")
% set(gca, 'FontSize', 15);

%% model evaluation
% disp("evaluating trained model...")
f1Max = [10,40];
tSpan = [0,10];
predIntervel = 10;
numCase = 50;
numTime = 100;
avgRMSE = evaluate_model_with_4_states(modelFile, sysParams, ctrlParams, trainParams, f1Max, tSpan, predIntervel, numCase, numTime, trainParams.type);
disp(["average rmse", avgRMSE])

%% plot single prediction
disp("plot prediction..."+modelFile)
f1Max = 25;
tSpan = [0,10];
predIntervel = 10;
plot_prediction(modelFile, sysParams, ctrlParams, trainParams, f1Max, tSpan, predIntervel, trainParams.type);

%% plot comparision
% folder = "model";
% typeList = ["dnn4","lstm4","pinn4","dnn6"];
% trainParams.numSamples = 50;
% numState = 4;
% f1Max = 25;
% tSpan = [0,10];
% predInterval = 10;
% numTime = 100;
% res = compare_model(folder, typeList, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, numTime, numState);