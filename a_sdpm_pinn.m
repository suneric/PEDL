%%
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

close all;
clear; 
clc;

%% settings
tSpan = [0,10];
tForceStop = 1;
ctrlOptions = control_options();

ds = load('trainingData.mat');
numSamples = length(ds.samples);
modelFile = "model/pinn_"+num2str(ctrlOptions.alpha)+"_"+num2str(numSamples)+".mat";
maxEpochs = 50;

%% generate data
% Feature data: 4-D initial state x0 + time interval
% the label data is a predicted state x=[q1,q2,q1dot,q2dot]
initTimes = 1:4; %start from 1 sec to 4 sec with 0.5 sec step 
xTrain = [];
yTrain = [];
for i = 1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:7,:); % q1,q2,q1_dot,q2_dot
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

%% make dnn and train 
numStates = 4; % q1,q2,q1dot,q2dot
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
    fullyConnectedLayer(numStates)];

% convert the layer array to a dlnetwork object
net = dlnetwork(layers);
% disp(net); 
% plot(net)

% training options
monitor = trainingProgressMonitor;
monitor.Metrics = "Loss";
monitor.Info = ["LearnRate","IterationPerEpoch","MaximumIteration","Epoch","Iteration"];
monitor.XLabel = "Epoch";

% Train the model using custom training loop
% Use the full data set at each iteration. Update the network learnable
% parameters and solver state using 'lbfgsupdate', at the end of each
% iteration, update the training progress monitor.
% create a function handle containing the loss for the L-BFGS update, and 
% use 'dlfeval' to evaluate the 'dlgradient' inside the modelLoss function 
% using automatic differentiation. 
% net = dlupdate(@double,net); % for better accuracy
% xTrain = dlarray(xTrain,'CB'); 
% yTrain = dlarray(yTrain,'CB');
% accfun = dlaccelerate(@modelLoss);
% lossFcn = @(net) dlfeval(accfun,net,xTrain,yTrain);
% solverState = lbfgsState; 
% for i = 1:maxEpochs
%     [net,solverState] = lbfgsupdate(net,lossFcn,solverState);
%     updateInfo(monitor,Epoch=i);
%     recordMetrics(monitor,i,TrainingLoss = solverState.Loss);
% end

% using stochastic gradient decent
miniBatchSize = 200;
learnRate = 0.001;
dataSize = size(yTrain,2);
numBatches = floor(dataSize/miniBatchSize);
numIterations = maxEpochs * numBatches;

momentum = 0.9; % for sgdmupdate
velocity = [];  % for sgdmupdate
averageGrad = [];
averageSqGrad = [];
iter = 0;
epoch = 0;
while epoch < maxEpochs && ~monitor.Stop
    epoch = epoch + 1;
    % Shuffle data.
    idx = randperm(dataSize);
    xTrain = xTrain(:,idx);
    yTrain = yTrain(:,idx);
    for j=1:numBatches
        iter  = iter + 1;
        startIdx = (j-1)*miniBatchSize+1;
        endIdx = min(j*miniBatchSize, dataSize);
        xBatch = xTrain(:,startIdx:endIdx);
        yBatch = yTrain(:,startIdx:endIdx); 
        X = gpuArray(dlarray(xBatch,"CB"));
        T = gpuArray(dlarray(yBatch,"CB"));
        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
        [loss,gradients] = dlfeval(@modelLoss,net,X,T);

        % Update the network parameters using the SGDM optimizer.
        % [net,velocity] = sgdmupdate(net,gradients,vel,learnRate,momentum);

        % Update the network parameters using the ADAM optimizer.
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iter,learnRate);

        recordMetrics(monitor,iter,Loss=loss);

        if mod(iter,maxEpochs) == 0
            updateInfo(monitor,LearnRate=learnRate,Epoch=epoch,Iteration=iter,MaximumIteration=numIterations,IterationPerEpoch=numBatches);
            monitor.Progress = 100*iter/numIterations;
        end
    end
end
save(modelFile,"net");

%% plot training loss and RMSE
figure('Position',[500,100,800,400]); 
tiledlayout("vertical","TileSpacing","tight")
info = monitor.MetricData.Loss;
x = info(:,1);
y = info(:,2);
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
x = y(:,4:7);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),4);
for i = 1:length(tp)
    xp(i,:) = extractdata(predict(net,dlarray([x0,tp(i)-t0]','CB')));
end
plot_compared_states(t,x,tp,xp)

%% Test 2
% simulation with small time interval
predInterval = 3;
net = load(modelFile).net;
ctrlOptions.fMax = [8;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:7);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
x0 = x(initIdx,:);
% prediction
tp = t(initIdx+1:end);
xp = zeros(length(tp),4);
for i = 1:length(tp)
    if (tp(i)-t0) > predInterval
        t0 = tp(i-1);
        x0 = xp(i-1,:);
    end
    xp(i,:) = extractdata(predict(net,dlarray([x0,tp(i)-t0]','CB')));
end
plot_compared_states(t,x,tp,xp)

%% loss function
function [loss, gradients, state] = modelLoss(net,X,T)
    % make prediction
    [Y, state] = forward(net,X);
    dataLoss = l2loss(Y,T);
   
    % compute gradients using automatic differentiation
    q1 = Y(1,:);
    q2 = Y(2,:);
    q1d = Y(3,:);
    q2d = Y(4,:);
    q1X = dlgradient(sum(q1,'all'), X);
    q2X = dlgradient(sum(q2,'all'), X);
    q1dX = dlgradient(sum(q1d,'all'), X);
    q2dX = dlgradient(sum(q2d,'all'), X);
    q1d = q1X(5,:);
    q2d = q2X(5,:); 
    q1dd = q1dX(5,:);
    q2dd = q2dX(5,:);
    f = physics_law([q1;q2],[q1d;q2d],[q1dd;q2dd]);
    zeroTarget = zeros(size(f),"like",f);
    physicLoss = l2loss(f,zeroTarget);
    
    % total loss
    ctrlOptions = control_options();
    loss = (1.0-ctrlOptions.alpha)*dataLoss + ctrlOptions.alpha*physicLoss;
    gradients = dlgradient(loss, net.Learnables);
end