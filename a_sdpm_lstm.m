%%
close all;
clear; 
clc;

%% set task type
tSpan = [0,10];
seqSteps = 10;
tForceStop = 1;
ctrlOptions = control_options();

ds = load('trainingData.mat');
numSamples = size(ds.samples,1);
modelFile = "model/lstm_"+num2str(ctrlOptions.alpha)+"_"+num2str(numSamples)+".mat";
maxEpochs = 50;

%% preprocess data for training
% Refer to the Help "Import Data into Deep Network Designer / Sequences and time series" 
initTimes = 1:4; %start from 1 sec to 4 sec with 0.5 sec step 
states = {};
times = [];
labels = [];
for i=1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:9,:);
    for tInit = initTimes
        initIdx = find(t >= tInit, 1, 'first');
        startIdx = initIdx-seqSteps+1;
        t0 = t(initIdx);
        x0 = [t(startIdx:initIdx);x(:,startIdx:initIdx)];
        for j=initIdx+1:length(t)
            states{end+1} = x0;
            times = [times,t(j)-t0];
            labels = [labels,x(:,j)];
        end
    end
end
disp([num2str(length(times)),' samples are generated for training.'])
states = reshape(states,[],1);
times = times';
labels = labels';

% split dataset into train and test datasets
% indices = randperm(size);
% num_train = round(size*training_percent);
% train_indices = indices(1:num_train);
% test_indices = indices(num_train+1:end);



%% Create Neural Network and Train
numStates = 6; % the 6-dim states of the predicted time step 
layers = [
    sequenceInputLayer(numStates+1)
    lstmLayer(32,OutputMode="last")
    concatenationLayer(1,2,Name="cat")
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
lgraph = addLayers(lgraph,[...
    featureInputLayer(1,Name="time")]);
lgraph = connectLayers(lgraph,"time","cat/in2");
% plot(lgraph);

% combine a datastore for training
miniBatchSize = 200;
dsState = arrayDatastore(states,'OutputType',"same",'ReadSize',miniBatchSize);
dsTime = arrayDatastore(times,'ReadSize',miniBatchSize);
dsLabel = arrayDatastore(labels,'ReadSize',miniBatchSize);
dsTrain = combine(dsState, dsTime, dsLabel);
% read(dsTrain)

options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    SequencePaddingDirection="left", ...
    Shuffle='every-epoch', ...
    Plots='training-progress', ...
    MiniBatchSize=miniBatchSize, ...
    Verbose=1);

% training with data store
[net,info] = trainNetwork(dsTrain,lgraph,options);
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
startIdx = initIdx-seqSteps+1;
state = {[t(startIdx:initIdx),x(startIdx:initIdx,:)]'};
x0 = arrayDatastore(state,'OutputType',"same",'ReadSize',miniBatchSize);
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp) 
    xp(i,:) = predict(net,combine(x0, arrayDatastore(tp(i)-t0,'ReadSize',miniBatchSize)));
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
startIdx = initIdx-seqSteps+1;
state = {[t(startIdx:initIdx),x(startIdx:initIdx,:)]'};
x0 = arrayDatastore(state,'OutputType',"same",'ReadSize',miniBatchSize);
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp) 
    if (tp(i)-t0) >= predInterval
        t0 = tp(i-1);
        initIdx = i-1;
        startIdx = initIdx-seqSteps+1;
        state = {[tp(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
        x0 = arrayDatastore(state,'OutputType',"same",'ReadSize',miniBatchSize);
    end
    xp(i,:) = predict(net,combine(x0, arrayDatastore(tp(i)-t0,'ReadSize',miniBatchSize)));
end
plot_compared_states(t,x,tp,xp)
