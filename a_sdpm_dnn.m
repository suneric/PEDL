%% Description
% DNN
% A DNN model for learning dynamics system behvior

close all;
clear; 
clc;

%% settings
tSpan = [0,10];
tForceStop = 1;
ctrlOptions = control_options();

ds = load('trainingData.mat');
numSamples = length(ds.samples);
modelFile = "model/dnn_"+num2str(ctrlOptions.alpha)+"_"+num2str(numSamples)+".mat";
maxEpochs = 50;

%% generate data
% Feature data: 6-D initial state x0 + time interval
% the label data is a predicted state x=[q1,q2,q1dot,q2dot,q1ddot,q2ddot]
initTimes = 1:4; %start from 1 sec to 4 sec with 0.5 sec step 
xTrain = [];
yTrain = [];
for i = 1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:9,:); % q1,q2,q1_dot,q2_dot
    for tInit = initTimes
        initIdx = find(t >= tInit,1,'first');
        x0 = x(:,initIdx); % Initial state 
        t0 = t(initIdx); % Start time
        for j = initIdx+1:length(t)
            xTrain = [xTrain,[x0; t(j)-t0]];
            yTrain = [yTrain,x(:,j)];
        end
    end
end
disp([num2str(length(xTrain)),' samples are generated for training.'])
xTrain = xTrain';
yTrain = yTrain';

%% Create Neural Network and Train
numStates = 6; % 6-dim states in the first second
layers = [
    featureInputLayer(numStates+1)
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numStates)
    myRegressionLayer("mse")];
lgraph = layerGraph(layers);
% plot(lgraph);

miniBatchSize = 200;
options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    Shuffle='every-epoch', ...
    Plots='training-progress', ...
    MiniBatchSize=miniBatchSize, ...
    Verbose=1);

% training with numeric array data
[net,info] = trainNetwork(xTrain,yTrain,lgraph,options);
save(modelFile,"net");
% disp(info)

%% plot training loss and RMSE
figure('Position',[500,100,800,400]); 
tiledlayout("vertical","TileSpacing","tight")
numIter = length(info.TrainingLoss);
x = 1:numIter;
y = info.TrainingLoss(x);
% z = info.ValidationRMSE(x);
smoothed_y = smoothdata(y,'gaussian');
% smoothed_z = movmean(z, window_size);
plot(x,y,'b-',x,smoothed_y,'r-',"LineWidth",2);
xlabel("Iteration","FontName","Arial")
ylabel("Loss","FontName","Arial")
legend("Original","Smoothed","location","best")
set(gca, 'FontSize', 15);

%% Test 1
net = load(modelFile).net;
ctrlOptions.fMax = [8;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp)
    xp(i,:) = predict(net,[x0,tp(i)-t0]);
end
plot_compared_states(t,x,tp,xp)

%% Test 2
% simulation with small time interval
predInterval = 3; 
net = load(modelFile).net;
ctrlOptions.fMax = [8;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp)
    if (tp(i)-t0) > predInterval
        t0 = tp(i-1);
        x0 = xp(i-1,:);
    end
    xp(i,:) = predict(net,[x0,tp(i)-t0]);
end
plot_compared_states(t,x,tp,xp)
