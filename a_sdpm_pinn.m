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
tForceStop = 1;
tSpan = [0,10];
ctrlOptions = control_options();

%% generate data
% Feature data: 4-D initial state x0 + time interval
% the label data is a predicted state x=[q1,q2,q1dot,q2dot]
xTrain = [];
yTrain = [];
ds = load('trainingData.mat');
for i = 1:length(ds.samples)
    data = load(ds.samples{i,1}).state;
    t = data(1,:);
    x = data(4:7,:); % q1,q2,q1_dot,q2_dot
    size = length(t);
    indices = find(t <= tForceStop);
    initIdx = indices(end)+1;
    x0 = x(:,initIdx); % Initial state 
    t0 = t(initIdx); % Start time
    for j = initIdx+1:size
        xTrain = [xTrain,[x0; t(j)-t0]];
        yTrain = [yTrain,x(1:4,j)];
    end
end
disp([num2str(length(xTrain)),' samples are generated for training.'])

%% make dnn and train 
numState = 4; % q1,q2,q1dot,q2dot
numLayers = 9;
numNeurons = 20;
layers = featureInputLayer(numState+1);
for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        reluLayer];
end
layers = [
    layers
    fullyConnectedLayer(numState)];

% convert the layer array to a dlnetwork object
net = dlnetwork(layers);
disp(net); 
% plot(net)
net = dlupdate(@double,net); % for better accuracy

% convert training data to formated dlarray
% 'C', channel, 'B', batch
xTrain = dlarray(xTrain,'CB'); 
yTrain = dlarray(yTrain,'CB');

% training options
numEpochs = 1500;
solverState = lbfgsState; 
% create a function handle containing the loss for the L-BFGS update, and 
% use 'dlfeval' to evaluate the 'dlgradient' inside the modelLoss function 
% using automatic differentiation. 
accfun = dlaccelerate(@modelLoss);
lossFcn = @(net) dlfeval(accfun,net,xTrain,yTrain);

monitor = trainingProgressMonitor;
monitor.Metrics = "TrainingLoss";
monitor.Info = ["LearningRate","Epoch","Iteration"];
monitor.XLabel = "Epoch";

% Train the model using custom training loop
% Use the full data set at each iteration. Update the network learnable
% parameters and solver state using 'lbfgsupdate', at the end of each
% iteration, update the training progress monitor.
for i = 1:numEpochs
    [net,solverState] = lbfgsupdate(net,lossFcn,solverState);
    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss = solverState.Loss);
end

fname = "pinn_model.mat";
save(fname,"net");

%% plot training loss and RMSE
figure('Position',[500,100,800,400]); 
tiledlayout("vertical","TileSpacing","tight")
info = monitor.MetricData.TrainingLoss;
x = info(:,1);
y = info(:,2);
% z = info.ValidationRMSE(x);
smoothed_y = smoothdata(y,'gaussian');
% smoothed_z = movmean(z, window_size);
plot(x,y,'b-',x,smoothed_y,'r-',"LineWidth",2);
xlabel("Iteration","FontName","Arial")
ylabel("TrainingLoss","FontName","Arial")
legend("Original","Smoothed","location","best")
set(gca, 'FontSize', 15);

%% Test
net = load("pinn_model.mat").net;
ctrlOptions.fMax = [3;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:7);
size = length(t);
indices = find(t <= tForceStop);
initIdx = indices(end)+1;
x0 = x(initIdx,:);
t0 = t(initIdx);
% prediction
xp = zeros(size,4);
xp(1:initIdx,:) = x(1:initIdx,:);
for i = initIdx+1:size
    xInit = dlarray([x0, t(i)-t0]','CB');
    xPred = predict(net,xInit);
    xp(i,:) = extractdata(xPred);
end
plot_states(t,x,xp,ctrlOptions)

%% loss function
function [loss, gradients] = modelLoss(net,X,T)
    % make prediction
    Y = forward(net,X);
    mseData = l2loss(Y,T);
    
    % compute gradients using automatic differentiation
    q1 = Y(1,:);
    q2 = Y(2,:);
    q1d = Y(3,:);
    q2d = Y(4,:); 
    t = X(5,:);
    q1dd = dlgradient(sum(q1d,'all'), t);
    q2dd = dlgradient(sum(q2d,'all'), t);
    % q1d = dlgradient(sum(q1,'all'), t);
    % q2d = dlgradient(sum(q2,'all'), t);

    % system parameters
    params = parameters();
    K = params(1);
    C = params(2);
    L = params(3);
    G = params(4);
    M1 = params(5);
    M2 = params(6);
    f1 = (M1+M2)*q1dd + M2*L*cos(q2).*q2dd + C*q1d + M2*L*sin(q2).*q2d.^2 - K*q1;
    f2 = M2*L*cos(q2).*q1dd + M2*L*L*q2dd + M2*G*L*sin(q2);
    f = [f1;f2];
    zeroTarget = zeros(size(f),"like",f);
    mseForce = l2loss(f,zeroTarget);
    
    alpha = 0.7;
    loss = (1-alpha)*mseForce + alpha*mseData;
    gradients = dlgradient(loss, net.Learnables);
end