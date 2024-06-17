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
xTrain = [];
yTrain = [];
for i = 1:numSamples
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:7,:); % q1,q2,q1_dot,q2_dot
    numTime = length(t);
    for tInit = 1:4
        indices = find(t <= tInit);
        initIdx = indices(end);
        x0 = x(:,initIdx); % Initial state 
        t0 = t(initIdx); % Start time
        for j = initIdx+1:numTime
            xTrain = [xTrain,[x0; t(j)-t0]];
            yTrain = [yTrain,x(1:4,j)];
        end
    end
end
disp([num2str(length(xTrain)),' samples are generated for training.'])

%% make dnn and train 
numStates = 4; % q1,q2,q1dot,q2dot
layers = [
    featureInputLayer(numStates+1)
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(128)
    tanhLayer
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
miniBatchSize = 128;
learnRate = 0.0001;
momentum = 0.9;
dataSize = size(yTrain,2);
numBatches = floor(dataSize/miniBatchSize);
numIterations = maxEpochs * numBatches;
vel = [];
iter = 0;
for i = 1:maxEpochs
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
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
        net.State = state;
        % Update the network parameters using the SGDM optimizer.
        [net,vel] = sgdmupdate(net,gradients,vel,learnRate,momentum);
        recordMetrics(monitor,iter,Loss=loss);
        updateInfo(monitor,LearnRate=learnRate,Epoch=i,Iteration=iter,MaximumIteration=numIterations,IterationPerEpoch=numBatches);
        monitor.Progress = 100*iter/numIterations;
        
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
numTime = length(t);
indices = find(t <= tForceStop);
initIdx = indices(end);
x0 = x(initIdx,:);
t0 = t(initIdx);
% prediction
xp = zeros(numTime,4);
xp(1:initIdx,:) = x(1:initIdx,:);
for i = initIdx+1:numTime
    xInit = dlarray([x0, t(i)-t0]','CB');
    xPred = predict(net,xInit);
    xp(i,:) = extractdata(xPred);
end
plot_compared_states(t,x,t,xp)

%% Test 2
% simulation with small time interval
predictTime = 3;
net = load(modelFile).net;
ctrlOptions.fMax = [8;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:7);
numTime = length(t);
indices = find(t <= tForceStop);
initIdx = indices(end);
x0 = x(initIdx,:);
t0 = t(initIdx);
% prediction
xp = zeros(numTime,4);
xp(1:initIdx,:) = x(1:initIdx,:);
for i = initIdx+1:numTime
    xInit = dlarray([x0, t(i)-t0]','CB');
    xPred = predict(net,xInit);
    xp(i,:) = extractdata(xPred);
    if (t(i)-t0) > predictTime
        t0 = t(i-1);
        x0 = xp(i-1,:);
    end
end
plot_compared_states(t,x,t,xp)

%% Test 3 
% simulation with small time step
net = load(modelFile).net;
ctrlOptions.fMax = [8;0];
tSpan = [0,10];
tForceStop = 1;
dTime = 0.01;
tic;
y = sdpm_simulation(tSpan, ctrlOptions);
t_ode = toc;
t = y(:,1);
x = y(:,4:7);
indices = find(t <= tForceStop);
initIdx = indices(end);
% predict with fixed time step
tPred = tForceStop+dTime:dTime:tSpan(end);
tp = zeros(length(tPred),1);
xp = zeros(length(tPred),4);
x0 = x(initIdx,:);
t0 = t(initIdx);
tic;
for i = 1:length(tPred)
    tp(i) = tPred(i);
    xInit = dlarray([x0,tPred(i)-t0]','CB');
    xPred = predict(net,xInit);
    xp(i,:) = extractdata(xPred);
end
t_dnn = toc;
disp(["ode:",t_ode]);
disp(["dnn:",t_dnn]);
plot_compared_states(t,x,tp,xp)

%% Test 4
% predict with small time interval from 1s to 5s
net = load(modelFile).net;
ctrlOptions.fMax = [4;0];
tForceStop = 1;
predictTime = 3; % time interval of prediction
dTime = 0.01; % time step of prediction
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:7);
indices = find(t <= tForceStop);
initIdx = indices(end);
% predict with fixed time step
tPred = tForceStop+dTime:dTime:tSpan(end);
tp = zeros(length(tPred),1);
xp = zeros(length(tPred),4);
x0 = x(initIdx,:);
t0 = t(initIdx);
% prediction
for i = 1:length(tPred)
    tp(i) = tPred(i);
    xInit = dlarray([x0,tPred(i)-t0]','CB');
    xPred = predict(net,xInit);
    xp(i,:) = extractdata(xPred);
    if (tPred(i)-t0 > predictTime)
        t0 = tPred(i-1);
        x0 = xp(i-1,:);
    end
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
    q1dX = dlgradient(sum(q1d,'all'), X);
    q1dd = q1dX(5,:);
    q2dX = dlgradient(sum(q2d,'all'), X);
    q2dd = q2dX(5,:);
    q1X = dlgradient(sum(q1,'all'), X);
    q1d = q1X(5,:);
    q2X = dlgradient(sum(q2,'all'), X);
    q2d = q2X(5,:);
    f = physics_law([q1;q2],[q1d;q2d],[q1dd;q2dd]);
    zeroTarget = zeros(size(f),"like",f);
    physicLoss = l2loss(f,zeroTarget);
    
    % total loss
    ctrlOptions = control_options();
    loss = (1.0-ctrlOptions.alpha)*dataLoss + ctrlOptions.alpha*physicLoss;
    gradients = dlgradient(loss, net.Learnables);
end