%%
close all;
clear; 
clc;

%% set task type
lossType = "PcNN"; % PgNN, PcNN, PiNN, change alpha in myRegressionLayer.m
task = "predict_next";
% task = "predict_arbitrary";
seq_steps = 20;
t_force_stop = 1;
training_percent = 0.95;
max_epochs = 12;

%% preprocess data for training
% Refer to the Help "Import Data into Deep Network Designer / Sequences and time series" 
ds = load('trainingData.mat');
num_samples = size(ds.samples,1);

states = {};
times = [];
labels = [];
for i=1:num_samples
    data = load(ds.samples{i,1}).state;
    switch task
        case "predict_next"
            [n,state,time,label] = create_data_next(data,seq_steps,t_force_stop);
        otherwise
            [n,state,time,label] = create_data_arbitrary(data,seq_steps,t_force_stop);
    end
    for j=1:n
        states{end+1} = state{j};
        times = [times,time(j)];
        labels = [labels,label(:,j)];
    end
end
states = reshape(states,[],1);
times = times';
labels = labels';

size = length(times);
disp([num2str(size),' samples are generated for training.'])
% split dataset into train and test datasets
indices = randperm(size);
num_train = round(size*training_percent);
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);

% combine a datastore for training
dsState = arrayDatastore(states(train_indices),'OutputType',"same",'ReadSize',128);
dsTime = arrayDatastore(times(train_indices),'ReadSize',128);
dsLabel = arrayDatastore(labels(train_indices,:),'ReadSize',128);
dsTrain = combine(dsState, dsTime, dsLabel);
% read(dsTrain)

dsState = arrayDatastore(states(test_indices),'OutputType',"same",'ReadSize',128);
dsTime = arrayDatastore(times(test_indices),'ReadSize',128);
dsLabel = arrayDatastore(labels(test_indices,:),'ReadSize',128);
dsTest = combine(dsState, dsTime, dsLabel);

%% Create Neural Network and Train
numFeatures = 9; % 6-dim states + time + 2 forces in the first second
numTime = 1; % the time step for prediction
numOutput = 6; % the 4-dim states of the predicted time step 

layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(64,OutputMode="last")
    fullyConnectedLayer(32)
    reluLayer
    concatenationLayer(1,2,Name="cat")
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(numOutput)
    myRegressionLayer("mse")];
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph,[...
    featureInputLayer(numTime)...
    fullyConnectedLayer(8,Name="time")]);
lgraph = connectLayers(lgraph,"time","cat/in2");
% plot(lgraph);

options = trainingOptions("adam", ...
    InitialLearnRate=0.0001, ...
    MaxEpochs=max_epochs, ...
    SequencePaddingDirection="left", ...
    Shuffle='every-epoch', ...
    Plots='training-progress', ...
    ValidationData=dsTest, ...
    MiniBatchSize=128, ...
    Verbose=1);

[net,info] = trainNetwork(dsTrain,lgraph,options);

fname = "model/"+lossType+"_model_"+num2str(num_samples)+".mat";
save(fname,"net");
disp(info)

%% plot training loss and RMSE
figure('Position',[500,100,800,400]); 
tiledlayout("vertical","TileSpacing","tight")
x = 1:5000;
y = info.TrainingRMSE(x);
% z = info.ValidationRMSE(x);
smoothed_y = smoothdata(y,'gaussian');
% smoothed_z = movmean(z, window_size);
plot(x,y,'b-',x,smoothed_y,'r-',"LineWidth",2);
xlabel("Iteration","FontName","Arial")
ylabel("RMSE","FontName","Arial")
legend("Original","Smoothed","location","best")
set(gca, 'FontSize', 15);
