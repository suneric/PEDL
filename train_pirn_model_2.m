function [modelFile, trainLoss] = train_pirn_model_2(sampleFile, trainParams)
% PINN
% A physics-Informed Neural Network (PINN) is a type of neural network
% architecture desigend to incorporate physical principles or equations
% into the learning process. In combines deep learning techniques with
% domain-specific knowledge, making it particularly suitable for problems
% governed by physics.
% In addition to standard data-driven training, PINNs utilize terms in the
% loss function to enforce consistency with know physical law, equations,
% and constraints. 
% https://en.wikipedia.org/wiki/Physics-informed_neural_networks 
% https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/
    
    % load samples and prepare training dataset
    ds = load(sampleFile);
    numSamples = length(ds.samples);    
    modelFile = "model\"+trainParams.type+"_"+num2str(trainParams.alpha)+"_"+num2str(numSamples)+".mat";
    
    % generate data
    % Feature data: 6-D initial state x0 + time interval
    % the label data is a predicted state x=[q1,q2,q1dot,q2dot,q1ddot,q2ddot]
    initTimes = 1:4; %start from 1 sec to 4 sec with 0.5 sec step 
    tTrain = [];
    xTrain = [];
    yTrain = [];
    for i = 1:numSamples
        data = load(ds.samples{i,1}).state;
        t = data(1,:);
        x = data(2:3, :); % q1,q2
        for tInit = initTimes
            initIdx = find(t > tInit, 1, 'first');
            x0 = x(:, initIdx); % Initial state 
            t0 = t(initIdx); % Start time
            for j = initIdx+1 : length(t)
                tTrain = [tTrain, t(j)-t0];
                xTrain = [xTrain, x0];
                yTrain = [yTrain, x(:,j)];
            end
        end
    end
    disp(num2str(length(xTrain)) + " samples are generated for training.");
    
    % Create neural network
    numStates = 2;
        inputLayer = featureInputLayer(numStates+1, "Name", "input");
    outputLayer = fullyConnectedLayer(numStates, "Name", "output");
    
    lgraph = layerGraph(inputLayer);
    lgraph = addLayers(lgraph, fullyConnectedLayer(trainParams.numNeurons, "Name", "fc_input"));
    lgraph = connectLayers(lgraph, "input", "fc_input");
    for i = 1:trainParams.numLayers
        resNetBlock = createSimpleResNetBlock(trainParams.numNeurons, i);
        lgraph = addLayers(lgraph, resNetBlock.Layers);
        if i == 1
            lgraph = connectLayers(lgraph,"fc_input", ['fc1_' num2str(i)]);
            lgraph = connectLayers(lgraph, "fc_input", ['add_' num2str(i) '/in2']);
        else
            lgraph = connectLayers(lgraph, ['add_' num2str(i-1)], ['fc1_' num2str(i)]);
            lgraph = connectLayers(lgraph, ['add_' num2str(i-1)], ['add_' num2str(i) '/in2']);
        end
    end
    lgraph = addLayers(lgraph, outputLayer);
    lgraph = connectLayers(lgraph, ['add_',num2str(trainParams.numLayers)], "output");

    % convert the layer array to a dlnetwork object
    net = dlnetwork(lgraph);
    net = dlupdate(@double, net);
    
    % training options
    monitor = trainingProgressMonitor;
    monitor.Metrics = "Loss";
    monitor.Info = ["LearnRate" ... 
                    "IterationPerEpoch" ...
                    "MaximumIteration" ...
                    "Epoch" ...
                    "Iteration" ...
                    "GradientsNorm"...
                    "StepNorm"];
    monitor.XLabel = "Iteration";
    
    net = train_adam_update(net, tTrain, xTrain, yTrain, trainParams, monitor);
    trainLoss = monitor.MetricData.Loss(:,2);
    
    save(modelFile, 'net');
end

%%
function net = train_adam_update(net, tTrain, xTrain, yTrain, trainParams, monitor)
    % using stochastic gradient decent
    miniBatchSize = trainParams.miniBatchSize;
    lrRate = trainParams.learningRate;
    dataSize = length(xTrain);
    numBatches = floor(dataSize/miniBatchSize);
    numIterations = trainParams.numEpochs * numBatches;

    accFcn = dlaccelerate(@modelLoss);
    
    avgGrad = [];
    avgSqGrad = [];
    iter = 0;
    epoch = 0;
    while epoch < trainParams.numEpochs && ~monitor.Stop
        epoch = epoch + 1;
        % Shuffle data.
        idx = randperm(dataSize);
        tAll = tTrain(:, idx);
        xAll = xTrain(:, idx);
        yAll = yTrain(:, idx);

        for j = 1 : numBatches
            iter = iter + 1;
            startIdx = (j-1)*miniBatchSize + 1;
            endIdx = min(j*miniBatchSize, dataSize);
            tBatch = tAll(:, startIdx:endIdx);
            xBatch = xAll(:, startIdx:endIdx);
            yBatch = yAll(:, startIdx:endIdx); 
            T = gpuArray(dlarray(tBatch, "CB"));
            X = gpuArray(dlarray(xBatch, "CB"));
            Y = gpuArray(dlarray(yBatch, "CB"));

            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss, gradients] = dlfeval(accFcn, net, T, X, Y);
    
            % Update the network parameters using the ADAM optimizer.
            [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, avgGrad, avgSqGrad, iter, lrRate);
    
            recordMetrics(monitor, iter, Loss=loss);
    
            if mod(iter, trainParams.numEpochs) == 0
                monitor.Progress = 100*iter/numIterations;
                updateInfo(monitor, ...
                    LearnRate = lrRate, ...
                    Epoch = epoch, ...
                    Iteration = iter, ...
                    MaximumIteration = numIterations, ...
                    IterationPerEpoch = numBatches);
            end
        end
    end
end

%% loss function
function [loss, gradients] = modelLoss(net, T, X, Y)
    % make prediction
    [Z, ~] = forward(net, [X;T]);
    dataLoss = mse(Z, Y);
    
    % compute gradients using automatic differentiation
    q1 = Z(1,:);
    q2 = Z(2,:);
    q1d = dlgradient(sum(q1, 'all'), T, EnableHigherDerivatives=true);
    q2d = dlgradient(sum(q2, 'all'), T, EnableHigherDerivatives=true);
    q1dd = dlgradient(sum(q1d, 'all'), T, EnableHigherDerivatives=true);
    q2dd = dlgradient(sum(q2d, 'all'), T, EnableHigherDerivatives=true);
    fY = physics_law([q1;q2], [q1d;q2d], [q1dd;q2dd]);
    fTarget = zeros(size(fY), 'like', fY);
    physicLoss = mse(fY, fTarget);
    
    global trainParams
    loss = (1.0-trainParams.alpha)*dataLoss + trainParams.alpha*(physicLoss);
    gradients = dlgradient(loss, net.Learnables);
end

function lgraph = createSimpleResNetBlock(numHiddenUnits, blockIndex)
    % Define layers for the ResNet block
    layers = [
        fullyConnectedLayer(numHiddenUnits, 'Name', ['fc1_' num2str(blockIndex)])
        tanhLayer('Name', ['tanh_' num2str(blockIndex)])
        fullyConnectedLayer(numHiddenUnits, 'Name', ['fc2_' num2str(blockIndex)])
        additionLayer(2,"Name", ['add_' num2str(blockIndex)])
    ];
    lgraph = layerGraph(layers);
end
