%% 
close all;
clear; 
clc;

sysParams = params_system();
ctrlParams = params_control();
ctrlParams.friction = "andersson";
ctrlParams.fixedTimeStep = 1e-4;
ctrlParams.method = "random";
ctrlParams.numPoints = 220;
% ctrlParams.method = "interval";
% ctrlParams.interval = 2e-2;

global trainParams;
trainParams = params_training();
trainParams.numSamples = 100;
trainParams.type = "dnn6";
trainParams.numEpochs = 1000;
trainParams.numLayers = 5; 
trainParams.numNeurons = 256;
trainParams.initLearningRate = 1e-3;
trainParams.stopLearningRate = 1e-5;
trainParams.miniBatchSize = 3000;
trainParams.lrDropEpoch = 5;

% generate f1, for training and test
numTestSamples = 100;
f1File = "f1Max.mat";
f1Max = [15,35];
if exist(f1File, 'file') == 2
    ds = load(f1File);
    disp(ds);
    f1Train = ds.f1Train;
    f1Test = ds.f1Test;
else
    numTotalSamples = trainParams.numSamples + numTestSamples;
    % f1Random = f1Max(1) + (f1Max(2)-f1Max(1))*rand(numTotalSamples,1);
    % f1Random = f1Random(randperm(numTotalSamples));
    % trainIndices = 1:trainParams.numSamples;
    % testIndices = trainParams.numSamples:numTotalSamples;

    f1Random = f1Max(1):(f1Max(2)-f1Max(1))/(numTotalSamples-1):f1Max(2);
    indices = 1:numTotalSamples;
    trainIndices = round(linspace(1,numTotalSamples,trainParams.numSamples));
    testIndices = setdiff(indices, trainIndices);
    f1Train = f1Random(trainIndices);
    f1Test = f1Random(testIndices);
    save(f1File,'f1Train','f1Test');
    
    % f1MaxTrain = [20,30];
    % f1Train = f1MaxTrain(1):(f1MaxTrain(2)-f1MaxTrain(1))/(trainParams.numSamples-1):f1MaxTrain(2);
    % 
    % f1Maxtest = [15,35];
    % f1Test = f1Maxtest(1) + (f1Maxtest(2)-f1Maxtest(1))*rand(numTestSamples,1);
    % save(f1File,'f1Train','f1Test');
end

% disp(f1Train);
% disp(f1Test);
% histogram(f1Train,trainParams.numSamples/2)
% histogram(f1Test,numTestSamples)

%% plot system motion with a sample
% ctrlParams.method = "origin";
% f1Max = 25;
% tSpan = [0,6];
% numState = 4;
% len = plot_system(sysParams, ctrlParams, f1Max, tSpan, numState);
% disp(num2str(len)+" time steps");

%% plot friction
% v = linspace(-2,2,500);
% plot_friction(sysParams, v);

%% generate samples
if ~exist("\data\", 'dir')
   mkdir("data");
end
tSpan = [0,6];
dataFile = generate_samples(sysParams, ctrlParams, trainParams, f1Train, tSpan);

%% train model
if ~exist("\model\", 'dir')
   mkdir("model");
end
switch trainParams.type
    case "dnn2"
        [modelFile, loss] = train_dnn_model_2(dataFile, trainParams);
    case "lstm2"
        [modelFile, loss] = train_lstm_model_2(dataFile, trainParams);
    case "pinn2"
        [modelFile, loss] = train_pinn_model_2(dataFile, trainParams);
    case "dnn4"
        [modelFile, loss] = train_dnn_model_4(dataFile, trainParams);
    case "lstm4"
        [modelFile, loss] = train_lstm_model_4(dataFile, trainParams);
    case "pinn4"
        [modelFile, loss] = train_pinn_model_4(dataFile, trainParams);
    case "dnn6"
        [modelFile, loss] = train_dnn_model_6(dataFile, trainParams);
    case "lstm6"
        [modelFile, loss] = train_lstm_model_6(dataFile, trainParams);
    case "pinn6"
        [modelFile, loss] = train_pinn_model_6(dataFile, trainParams);
    case "pirn2"
        [modelFile, loss] = train_pirn_model_2(dataFile, trainParams);
    case "pirn4"
        [modelFile, loss] = train_pirn_model_4(dataFile, trainParams);
    case "pirn6"
        [modelFile, loss] = train_pirn_model_6(dataFile, trainParams);
    otherwise
        disp("unspecified type of model.")
end
lossFile = replace(modelFile, '.mat', '_loss.mat');
save(lossFile, 'loss');

%% model evaluation
disp("evaluating trained model...")
tSpan = [1,6];
predIntervel = 8;
numTime = 150;
ctrlParams.fixedTimeStep = 0;
ctrlParams.method = "origin";
net = load(modelFile).net;
errs = evaluate_model_with_4_states(net, sysParams, ctrlParams, trainParams, f1Test, tSpan, predIntervel, numTime, trainParams.type);
errFile = replace(modelFile, '.mat', '_err.mat');
save(errFile, 'errs');
avgErr = mean(errs,'all'); % one value of error for estimtation
disp(["average rmse", avgErr])

%% plot single prediction
f1Max = 32;
ctrlParams.fixedTimeStep = 0;
ctrlParams.method = "origin";
tSpan = [1,6];
predIntervel = 8;
numState = 4;  
trainParams.type = "dnn6";
modelFile = "model_best\"+trainParams.type+"_"+num2str(trainParams.numSamples)+".mat";
disp("plot prediction..."+modelFile)
net = load(modelFile).net;
[t,x,xp] = plot_prediction(net, sysParams, ctrlParams, trainParams, f1Max, tSpan, predIntervel, trainParams.type, numState);
tSnapshot = [1,6];
% sdpm_snapshot(sysParams,t,x(:,2),x(:,3),xp(:,2),xp(:,3),tSnapshot);
% sdpm_animation(sysParams,t,x(:,2),x(:,3),xp(:,2),xp(:,3),tSnapshot);

%% plot training curves
dnnLoss = load("model\dnn4_100_loss.mat").loss;
lstmLoss = load("model\lstm4_100_loss.mat").loss;
pinnLoss = load("model\pinn4_100_loss.mat").loss;
pgnnLoss = load("model\dnn6_100_loss.mat").loss;

figure('Position',[500,100,500,400]); 
iter = 1:10000;
% smoothdata(dnnLoss(iter),'gaussian')
plot(iter,dnnLoss(iter), "LineWidth",2,"DisplayName","MLP");
hold on;
plot(iter,lstmLoss(iter), "LineWidth",2,"DisplayName","LSTM","LineStyle",":");
hold on;
plot(iter,pinnLoss(iter), "LineWidth",2,"DisplayName","PINN","LineStyle","--");
hold on;
plot(iter,pgnnLoss(iter), "LineWidth",2,"DisplayName","PGNN","LineStyle","-.");
xlabel("Iteration","FontName","Arial")
ylabel("Loss","FontName","Arial")
legend("location","northeast","FontName","Arial")
title("Training Loss","FontName","Arial");
set(gca, 'FontSize', 15);

%% plot comparision
folder = "model";
typeList = ["dnn4","lstm4","pinn4","dnn6"];
trainParams.numSamples = 100;
numState = 4;
f1Max = 25;
tSpan = [1,9];
predInterval = 10;
numTime = 150;
res = compare_model(folder, typeList, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, numTime, numState);

%% plot average error
dnnErr = load("model\dnn4_100_err.mat").errs;
lstmErr = load("model\lstm4_100_err.mat").errs;
pinnErr = load("model\pinn4_100_err.mat").errs;
pgnnErr = load("model\dnn6_100_err.mat").errs;

figure('Position',[500,100,800,400],'Color','White');
t = linspace(0,5,150);
plot(t,mean(dnnErr,1), "LineWidth",2,"DisplayName","MLP");
hold on
plot(t,mean(lstmErr,1), "LineWidth",2,"DisplayName","LSTM","LineStyle",":");
hold on
plot(t,mean(pinnErr,1), "LineWidth",2, "DisplayName","PINN","LineStyle","--");
hold on
plot(t,mean(pgnnErr,1), "LineWidth",2,"DisplayName","PGNN","LineStyle","-.");
title("Average Prediction Errors");
xlabel("Time (s)");
ylabel("RMSE");
legend("Location","northeast");
set(gca,"FontName","Arial", "FontSize", 15);

%% prediction time
% ctrlParams.fMax = [20; 0];
% x0 = zeros(4,1);
% tSpan = [0,1];
% % tic;
% [t,x] = ode45(@(t,x) sdpm_system(t, x, sysParams, ctrlParams), tSpan, x0);
% % t_sim = toc;
% % fprintf('Elapsed time is %.3f seconds.\n', t_sim);
% 
% numTime = length(t);
% y = zeros(numTime,4);
% for i = 1:numTime
%     F = force_function(t(i), ctrlParams);
%     fc = coulomb_friction(x(i,2), sysParams, ctrlParams.friction);
%     xdot = compute_xdot(x(i,:), F, fc, sysParams);
%     y(i,1) = x(i, 1); % q1
%     y(i,2) = x(i, 3); % q2
%     y(i,3) = x(i, 2); % q1dot
%     y(i,4) = x(i, 4); % q2dot
%     y(i,5) = xdot(2); % q1ddot
%     y(i,6) = xdot(4); % q2ddot
% end
% 
% tp = 5;
% % MLP
% % net = load("model\dnn4_100.mat").net;
% % x0 = y(end,1:4)
% 
% % LSTM
% % net = load("model\lstm4_100.mat").net;
% % x0 = {[t(end-8:end), y(end-8:end,1:4)]'};
% % dsState = arrayDatastore(x0, 'OutputType', 'same', 'ReadSize',1);
% % dsTime = arrayDatastore(tp, 'ReadSize', 1);
% 
% % PINN
% % net = load("model\pinn4_100.mat").net;
% % x0 = y(end,1:4);
% 
% % PGNN
% net = load("model\dnn6_100.mat").net;
% x0 = y(end,:);
% 
% tic;
% % xp = predict(net, [x0, tp]);
% % xp = predict(net, combine(dsState, dsTime));
% % xp = extractdata(predict(net, dlarray([x0, tp]', 'CB')));
% xp = predict(net, [x0, tp]);
% t_dl = toc;
% fprintf('Elapsed time is %.3f seconds.\n', t_dl);

