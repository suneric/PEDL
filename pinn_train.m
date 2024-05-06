%%
close all;
clear; 
clc;

%% generate data
tForceStop = 1;
x_train = [];
y_train = [];

ds = load('trainingData.mat');
num_samples = size(ds.samples,1);
for i = 1:num_samples
    data = load(ds.samples{1,1}).state;
    t = data(1,:);
    x = data(4:7,:);
    size = length(t);
    indices = find(t <= tForceStop);
    init_idx = indices(end);
    x0 = x(:,init_idx);
    for j = indices(end)+1:size
        x_train = [x_train,[x0;t(j)-t(indices(end))]];
        y_train = [y_train,x(:,j)];
    end
end
x_train = x_train';
y_train = y_train';

size = length(x_train);
disp([num2str(size),' samples are generated for training.'])

% dsInit = arrayDatastore(x0_train,'ReadSize',128);
% dsTime = arrayDatastore(t_train,'ReadSize',128);
% dsState = arrayDatastore(x_train,'ReadSize',128);
% dsTrain = combine(dsInit,dsTime,dsState);
% read(dsTrain)

%%
numState = 4; % q1,q2,q1dot,q2dot
numTime = 1;
layers = [
    featureInputLayer(numState+numTime)
    fullyConnectedLayer(32)
    fullyConnectedLayer(32)
    fullyConnectedLayer(numState)
    regressionLayer];
lgraph = layerGraph(layers);
% plot(lgraph);

options = trainingOptions('adam', ...
        MaxEpochs = 10, ...
        MiniBatchSiz = 32, ...
        Verbose = true, ...
        Plots = 'training-progress');
net = trainNetwork(x_train,y_train,lgraph,options);

%% Test
tSpan = [0,10];
ctrlOptions = control_options();
% max_forces = linspace(0.5,15,30);
max_forces = [3];
tTest = linspace(1,10,50);
num_test = length(max_forces);
err_list = zeros(6*num_test,length(tTest));
for i = 1:num_test
    disp("test case: "+num2str(i)+" / "+num2str(num_test));
    ctrlOptions.fMax = [max_forces(i);0];
    y = sdpm_simulation(tSpan,ctrlOptions);
    t = y(:,1);
    x = y(:,4:7);

    size = length(t);
    indices = find(t <= tForceStop);
    initIdx = indices(end);
    x0 = x(initIdx,:);
    % prediction
    x_pred = zeros(size,4);
    x_pred(1:initIdx,:) = x(1:initIdx,:);
    for j = indices(end)+1:size
        dTime = t(j)-t(indices(end));
        xTest = [x0,dTime];
        x_pred(j,:) = predict(net,xTest);
    end
    plot_states(t,x,ctrlOptions,x_pred)
end

function loss = physics_loss(x,t)
    % system parameters
    params = parameters();
    K = params(1);
    C = params(2);
    L = params(3);
    G = params(4);
    M1 = params(5);
    M2 = params(6);
    % compute derivatives
    dxdt = grdient(x,t);
    q1 = x(1);
    q2 = x(2);
    q1dot = x(3);
    q2dot = x(4);
    q1ddot = dxdt(3);
    q2ddot = dxdt(4);
    % Evaluate differential equation
    A = [M1+M2 M2*L*cos(q2); M2*L*cos(q2) M2*L*L];
    B = [C*q1dot+M2*L*sin(q2)*q2dot*q2dot-K*q1; M2*G*L*sin(q2)];
    eqn = A*[q1ddot;q2ddot] + B;
    loss = mean(eqn.^2);
end