function modelFile = train_pinn_model(sampleFile, trainParams)
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
    
    %% generate data
    % Feature data: 6-D initial state x0 + time interval
    % the label data is a predicted state x=[q1,q2,q1dot,q2dot,q1ddot,q2ddot]
    initTimes = 1:0.5:4; %start from 1 sec to 4 sec with 0.5 sec step 
    xTrain = [];
    yTrain = [];
    for i = 1:numSamples
        data = load(ds.samples{i,1}).state;
        t = data(1,:);
        x = data(2:7, :); % q1,q2,q1_dot,q2_dot,q1_ddot,q2_ddot
        for tInit = initTimes
            initIdx = find(t >= tInit, 1, 'first');
            x0 = x(:, initIdx); % Initial state 
            t0 = t(initIdx); % Start time
            for j = initIdx+1 : length(t)
                xTrain = [xTrain, [x0; t(j)-t0]];
                yTrain = [yTrain, x(:,j)];
            end
        end
    end
    disp(num2str(length(xTrain)) + " samples are generated for training.");
    
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
            reluLayer
        ];
    end
    layers = [
        layers
        dropoutLayer(trainParams.dropoutFactor)
        ];
    for i = numMiddle+1:trainParams.numLayers
        layers = [
            layers
            fullyConnectedLayer(trainParams.numNeurons)
            reluLayer
        ];
    end
    
    layers = [
        layers
        fullyConnectedLayer(numStates, "Name", "output")
       ];

    % convert the layer array to a dlnetwork object
    net = dlnetwork(layers);
    % plot(net)
    
    % training options
    monitor = trainingProgressMonitor;
    monitor.Metrics = "Loss";
    monitor.Info = ["LearnRate", "IterationPerEpoch", "MaximumIteration", "Epoch", "Iteration"];
    monitor.XLabel = "Epoch";
    
    % using stochastic gradient decent
    miniBatchSize = trainParams.miniBatchSize;
    lrRate = trainParams.learningRate;
    dataSize = size(yTrain,2);
    numBatches = floor(dataSize/miniBatchSize);
    numIterations = trainParams.numEpochs * numBatches;
    
    avgGrad = [];
    avgSqGrad = [];
    iter = 0;
    epoch = 0;
    while epoch < trainParams.numEpochs && ~monitor.Stop
        epoch = epoch + 1;
        % Shuffle data.
        idx = randperm(dataSize);
        xTrain = xTrain(:, idx);
        yTrain = yTrain(:, idx);
        for j = 1 : numBatches
            iter = iter + 1;
            startIdx = (j-1)*miniBatchSize + 1;
            endIdx = min(j*miniBatchSize, dataSize);
            xBatch = xTrain(:, startIdx:endIdx);
            yBatch = yTrain(:, startIdx:endIdx); 
            X = gpuArray(dlarray(xBatch, "CB"));
            T = gpuArray(dlarray(yBatch, "CB"));
            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss, gradients] = dlfeval(@modelLoss, net, X, T);
    
            % Update the network parameters using the ADAM optimizer.
            [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, avgGrad,avgSqGrad,iter,lrRate);
    
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
    save(modelFile, 'net');

end

%% loss function
function [loss, gradients, state] = modelLoss(net, X, T)
    % make prediction
    [Y, state] = forward(net, X);
    dataLoss = l2loss(Y, T);
    % dataLoss = mean((Y-T).^2, 'all');
   
    % compute gradients using automatic differentiation
    q1 = Y(1,:);
    q2 = Y(2,:);
    q1d = Y(3,:);
    q2d = Y(4,:);
    q1dX = dlgradient(sum(q1d, 'all'), X);
    q2dX = dlgradient(sum(q2d, 'all'), X);
    % q1X = dlgradient(sum(q1, 'all'), X);
    % q2X = dlgradient(sum(q2, 'all'), X);
    % q1d = q1X(7, :);
    % q2d = q2X(7, :);
    q1dd = q1dX(7, :);
    q2dd = q2dX(7, :);
    fT = physics_law(T(1:2, :), T(3:4, :), T(5:6, :));
    fY = physics_law([q1; q2], [q1d; q2d], [q1dd; q2dd]);
    physicLoss = l2loss(fY, fT);
    % physicLoss = mean((fY-fT).^2, 'all');
    
    % total loss
    global trainParams
    loss = (1.0-trainParams.alpha)*dataLoss + trainParams.alpha*physicLoss;
    gradients = dlgradient(loss, net.Learnables);
end
