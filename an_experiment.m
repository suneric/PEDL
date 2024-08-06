%% 
close all;
clear; 
clc;

global trainParams;
trainParams = params_training();
sysParams = params_system();
ctrlParams = params_control();
trainParams.type = "pinn6";

%% plot system motion with a sample
% f1Max = 20;
% tSpan = [0,5];
% len = plot_system(sysParams, ctrlParams, f1Max, tSpan);
% disp(num2str(len)+" time steps");

%% generate samples
if ~exist("\data\", 'dir')
   mkdir("data");
end
[dataFile, fMaxRange] = generate_samples(sysParams, ctrlParams, trainParams);
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
    otherwise
        disp("unspecified type of model.")
end

%% plot training curve
figure('Position',[500,100,800,400]); 
tiledlayout("vertical","TileSpacing","tight")
x = 1:length(trainLoss);
y = trainLoss(x);
smoothed_y = smoothdata(y,'gaussian');
plot(x,y,'b-',x,smoothed_y,'r-',"LineWidth",2);
xlabel("Iteration","FontName","Arial")
ylabel("Loss","FontName","Arial")
legend("Original","Smoothed","location","best")
set(gca, 'FontSize', 15);

%% model evaluation
% disp("evaluating trained model...")
tSpan = [0,10];
predIntervel = 10;
numCase = 30;
numTime = 60;
% avgRMSE = evaluate_model(modelFile, sysParams, ctrlParams, trainParams, tSpan, predIntervel, numCase, numTime, trainParams.type);
% disp(["average rmse", avgRMSE])

%% plot single prediction
disp("plot prediction..."+modelFile)
f1Max = 15;
tSpan = [0,10];
predIntervel = 10;
% plot_prediction(modelFile, sysParams, ctrlParams, trainParams, f1Max, tSpan, predIntervel, trainParams.type);

%% plot comparision
folder = "model_friction";
typeList = ["dnn6","lstm6","pinn6"];
trainParams.numSamples = 500;
numState = 6;
f1Max = 5;
tSpan = [0,10];
predInterval = 10;
numTime = 60;
% res = compare_model(folder, typeList, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, numTime, numState);