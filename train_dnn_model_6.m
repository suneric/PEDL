function [modelFile, trainLoss] = train_dnn_model_6(sampleFile, trainParams)
% Train a DNN model for learning dynamics system behvior
    % load samples and prepare training dataset
    ds = load(sampleFile);
    numSamples = length(ds.samples);    
    modelFile = "model\"+trainParams.type+"_"+num2str(trainParams.alpha)+"_"+num2str(numSamples)+".mat";
    
    % generate training dataset
    % Feature: 6-D initial state (x0) + the predict future time (t)
    % Label: a predicted state x = [q1,q2,q1dot,q2dot,q1ddot,q2ddot]'
    % Start from 1 sec to 4 sec with 0.5 sec step 
    initTimes = 1:trainParams.initTimeStep:4; 
    xTrain = [];
    yTrain = [];
    for i = 1:numSamples
        data = load(ds.samples{i,1}).state;
        t = data(1,:);
        x = data(2:7,:);
        for tInit = initTimes
            initIdx = find(t > tInit, 1, 'first');
            x0 = x(:,initIdx);  % Initial state 
            t0 = t(initIdx);    % Start time
            for j = initIdx+1:length(t)
                xTrain = [xTrain, [x0; t(j)-t0]];
                yTrain = [yTrain, x(:,j)];
            end
        end
    end
    disp(num2str(length(xTrain)) + " samples are generated for training.");
    xTrain = xTrain';
    yTrain = yTrain';

    % Create neural network
    numStates = 6;
    layers = [
        featureInputLayer(numStates+1, "Name", "input")
        ];
    
    numMiddle = floor(trainParams.numLayers/2);
    for i = 1:numMiddle
        layers = [
            layers
            fullyConnectedLayer(trainParams.numNeurons)
            tanhLayer
        ];
    end
    if trainParams.dropoutFactor > 0
        layers = [
            layers
            dropoutLayer(trainParams.dropoutFactor)
        ];
    end
    for i = numMiddle+1:trainParams.numLayers
        layers = [
            layers
            fullyConnectedLayer(trainParams.numNeurons)
            tanhLayer
        ];
    end
    
    layers = [
        layers
        fullyConnectedLayer(numStates, "Name", "output")
        weightedLossLayer("mse")
       ];

    lgraph = layerGraph(layers);
    % plot(lgraph);
    
    options = trainingOptions("adam", ...
        InitialLearnRate = trainParams.initLearningRate, ...
        MaxEpochs = trainParams.numEpochs, ...
        MiniBatchSize = trainParams.miniBatchSize, ...
        Shuffle = "every-epoch", ...
        Plots = "training-progress", ...
        Verbose = 1);
    
    % training with numeric array data
    [net,info] = trainNetwork(xTrain,yTrain,lgraph,options);
    trainLoss = info.TrainingLoss;
    save(modelFile, 'net');
    % disp(info)
end
    