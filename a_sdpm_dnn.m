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
maxEpochs = 20;

%% generate data
% Feature data: 6-D initial state x0 + time interval
% the label data is a predicted state x=[q1,q2,q1dot,q2dot,q1ddot,q2ddot]
xTrain = [];
yTrain = [];
for i = 1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:9,:); % q1,q2,q1_dot,q2_dot
    numTime = length(t);
    for tInit = 1:4 % using every second data as initial state
        indices = find(t <= tInit);
        initIdx = indices(end);
        x0 = x(:,initIdx); % Initial state 
        t0 = t(initIdx); % Start time
        for j = initIdx+1:numTime
            xTrain = [xTrain,[x0; t(j)-t0]];
            yTrain = [yTrain,x(1:6,j)];
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
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(numStates)
    myRegressionLayer("mse")];
lgraph = layerGraph(layers);
% plot(lgraph);

options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    Shuffle='every-epoch', ...
    Plots='training-progress', ...
    MiniBatchSize=128, ...
    Verbose=1);

% training with numeric array data
[net,info] = trainNetwork(xTrain,yTrain,lgraph,options);
save(modelFile,"net");
% disp(info)

%% plot training loss and RMSE
figure('Position',[500,100,800,400]); 
tiledlayout("vertical","TileSpacing","tight")
numIter = length(info.TrainingRMSE);
x = 1:numIter;
y = info.TrainingRMSE(x);
% z = info.ValidationRMSE(x);
smoothed_y = smoothdata(y,'gaussian');
% smoothed_z = movmean(z, window_size);
plot(x,y,'b-',x,smoothed_y,'r-',"LineWidth",2);
xlabel("Iteration","FontName","Arial")
ylabel("RMSE","FontName","Arial")
legend("Original","Smoothed","location","best")
set(gca, 'FontSize', 15);

%% Test 1
net = load(modelFile).net;
ctrlOptions.fMax = [8;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
numTime = length(t);
indices = find(t <= tForceStop);
initIdx = indices(end);
x0 = x(initIdx,:);
t0 = t(initIdx);
% prediction
xp = zeros(numTime,6);
xp(1:initIdx,:) = x(1:initIdx,:);
for i = initIdx+1:numTime
    xInit = [x0,t(i)-t0];
    xPred = predict(net,xInit);
    xp(i,:) = xPred;
end
plot_compared_states(t,x,t,xp)

%% Test 2
% simulation with small time interval
predictTime = 3; 
net = load(modelFile).net;
ctrlOptions.fMax = [8;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
numTime = length(t);
indices = find(t <= tForceStop);
initIdx = indices(end);
x0 = x(initIdx,:);
t0 = t(initIdx);
% prediction
xp = zeros(numTime,6);
xp(1:initIdx,:) = x(1:initIdx,:);
for i = initIdx+1:numTime
    xInit = [x0, t(i)-t0];
    xPred = predict(net,xInit);
    xp(i,:) = xPred;
    if (t(i)-t0) > predictTime
        t0 = t(i-1);
        x0 = xp(i-1,:);
    end
end
plot_compared_states(t,x,t,xp)
