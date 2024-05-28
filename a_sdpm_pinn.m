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
numState = 4; % q1,q2,q1dot,q2dot
numLayers = 4;
numNeurons = 128;
layers = featureInputLayer(numState+1);
for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
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
numEpochs = 5000;
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

%% Test 1
% simulation with different tForceStop
tForceStop = 1;
net = load("pinn_model.mat").net;
ctrlOptions.fMax = [6;0];
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
tForceStop = 1;
predictTime = 3;
net = load("pinn_model.mat").net;
ctrlOptions.fMax = [7;0];
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
net = load("pinn_model.mat").net;
ctrlOptions.fMax = [8;0];
tSpan = [0,10];
tForceStop = 1;
tic;
y = sdpm_simulation(tSpan, ctrlOptions);
t_ode = toc;
t = y(:,1);
x = y(:,4:7);
indices = find(t <= tForceStop);
initIdx = indices(end);
% predict with fixed time step
dTime = 0.01;
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
tForceStop = 1;
predictTime = 2;
dTime = 0.01;
net = load("pinn_model.mat").net;
ctrlOptions.fMax = [6;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:7);
numTime = length(t);
indices = find(t <= tForceStop);
initIdx = indices(end);

tPred = tForceStop+dTime:dTime:tSpan(end);
tp = zeros(length(tPred),1);
xp = zeros(length(tPred),4);
x0 = x(initIdx,:);
t0 = t(initIdx);

% prediction
xp = zeros(numTime,4);
xp(1:initIdx,:) = x(1:initIdx,:);
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
function [loss, gradients] = modelLoss(net,X,T)
    % make prediction
    Y = forward(net,X);
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
    loss = dataLoss + physicLoss;
    gradients = dlgradient(loss, net.Learnables);
end

function plot_compared_states(t,x,tp,xp)
    labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$"];
    figure('Position',[500,100,800,800]);
    tiledlayout("vertical","TileSpacing","tight")
    numState = size(xp);
    numState = numState(2);
    for i = 1:numState
        nexttile
        plot(t,x(:,i),'b-',tp,xp(:,i),'r--','LineWidth',2);
        hold on
        xline(1,'k--', 'LineWidth',1);
        ylabel(labels(i),"Interpreter","latex");
        set(get(gca,'ylabel'),'rotation',0);
        set(gca, 'FontSize', 15);
        set(gca, 'FontName', "Arial")
        if i == numState
            xlabel("Time (s)");
        end
    end 
    legend("Reference","Prediction","Location","best","FontName","Arial");
end