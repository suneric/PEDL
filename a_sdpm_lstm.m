%%
close all;
clear; 
clc;

%% set task type
tSpan = [0,10];
seqSteps = 20;
tForceStop = 1;
ctrlOptions = control_options();

ds = load('trainingData.mat');
numSamples = size(ds.samples,1);
modelFile = "model/lstm_"+num2str(ctrlOptions.alpha)+"_"+num2str(numSamples)+".mat";
maxEpochs = 20;

%% preprocess data for training
% Refer to the Help "Import Data into Deep Network Designer / Sequences and time series" 
states = {};
times = [];
labels = [];
for i=1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:9,:);
    numTime = length(t);
    for tInit = 1:4 % sample from 1 to 4 second
        indices = find(t <= tInit);
        initIdx = indices(end);
        startIdx = initIdx-seqSteps+1;
        t0 = t(initIdx);
        x0 = [t(1,startIdx:initIdx);x(:,startIdx:initIdx)];
        for j=initIdx+1:numTime
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

% combine a datastore for training
dsState = arrayDatastore(states,'OutputType',"same",'ReadSize',128);
dsTime = arrayDatastore(times,'ReadSize',128);
dsLabel = arrayDatastore(labels,'ReadSize',128);
dsTrain = combine(dsState, dsTime, dsLabel);
% read(dsTrain)

%% Create Neural Network and Train
numFeatures = 7; % 6-dim states + time
numTime = 1; % the time for prediction
numOutput = 6; % the 6-dim states of the predicted time step 

layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(64,OutputMode="last")
    fullyConnectedLayer(128)
    tanhLayer
    concatenationLayer(1,2,Name="cat")
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(numOutput)
    myRegressionLayer("mse")];
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph,[...
    featureInputLayer(numTime)...
    fullyConnectedLayer(16,Name="time")]);
lgraph = connectLayers(lgraph,"time","cat/in2");
% plot(lgraph);

options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    SequencePaddingDirection="left", ...
    Shuffle='every-epoch', ...
    Plots='training-progress', ...
    MiniBatchSize=128, ...
    Verbose=1);

% training with data store
[net,info] = trainNetwork(dsTrain,lgraph,options);
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
xp = zeros(numTime,6);
indices = find(t <= tForceStop);
xp(indices,:) = x(indices,:);
initIdx = indices(end);
startIdx = initIdx-seqSteps+1;
state = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
dsState = arrayDatastore(state,'OutputType',"same",'ReadSize',128);
t0 = t(initIdx);
for i = initIdx+1:numTime  
    dsTime = arrayDatastore(t(i)-t0,'ReadSize',128);
    dsTest = combine(dsState, dsTime);
    xp(i,:) = predict(net,dsTest);
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
xp = zeros(numTime,6);
indices = find(t <= tForceStop);
xp(indices,:) = x(indices,:);
initIdx = indices(end);
startIdx = initIdx-seqSteps+1;
t0 = t(initIdx);
state = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
dsState = arrayDatastore(state,'OutputType',"same",'ReadSize',128);
for i = initIdx+1:numTime
    dsTime = arrayDatastore(t(i)-t0,'ReadSize',128);
    dsTest = combine(dsState, dsTime);
    xp(i,:) = predict(net,dsTest);
    if (t(i)-t0) >= predictTime
        initIdx = i-1;
        startIdx = initIdx-seqSteps+1;
        t0 = t(initIdx);
        state = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
        dsState = arrayDatastore(state,'OutputType',"same",'ReadSize',128);
    end
end
plot_compared_states(t,x,t,xp)
