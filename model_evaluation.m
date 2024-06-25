%%
close all;
clear;
clc;
disp("clear and start the program")

%% set task type
params = parameters();
seqSteps = 10;
tForceStop = 1;% time stop force
tSpan = [0,10]; % simulation time span
ctrlOptions = control_options();
disp("initialize parameters");

modelType = "lstm"; % "dnn", "pinn", "lstm"
numSamples = 100;
modelFile = "model/"+modelType+"_"+num2str(ctrlOptions.alpha)+"_"+num2str(numSamples)+".mat";
net = load(modelFile).net;
predInterval = 3;
F1Min = max(20,params(10));

%% Single case prediction accuracy over specified time span
ctrlOptions.fMax = [F1Min+8;0];
y = sdpm_simulation(tSpan, ctrlOptions);
t = y(:,1);
x = y(:,4:9);
xp = predict_motion(net,modelType,t,x,predInterval,seqSteps,tForceStop);

tTest = [1,10];
indices = find(t >= tTest(1) & t <= tTest(end));
rse = root_square_err(indices,x,xp);
% disp(mean(rse,1));
disp("Single case predition accuracy")
figure('Position',[500,100,1000,800]);
labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
tiledlayout("vertical","TileSpacing","tight")
numState = size(xp);
for i = 1:numState(2)
    nexttile
    plot(t,x(:,i),'b-',t,xp(:,i),'k--','LineWidth',2);
    hold on
    xline(1,'k--', 'LineWidth',1);
    ylabel(labels(i),"Interpreter","latex");
    xticks([])
    if i == 6
        xlabel("Time (s)");
        xticks([1,2,3,4,5,6,7,8,9,10])
    end
    if i == 1
        legend("Reference","Prediction","Location","northeastoutside");
    end
    set(get(gca,'ylabel'),'rotation',0);
    set(gca, 'FontSize', 15);
    set(gca, 'FontName', "Arial");
end 

%% Prediction Accuracy evluation
% evaluate the model with specified forces, and time steps
numCase = 50;
numTime = 100;
refTime = linspace(1,10,numTime);
maxForces = linspace(0.5,15,numCase);
errs = zeros(4*numCase,numTime);
for i = 1:numCase
    % reference
    ctrlOptions.fMax = [F1Min+maxForces(i);0];
    y = sdpm_simulation(tSpan, ctrlOptions);
    t = y(:,1);
    x = y(:,4:9);
    xp = predict_motion(net,modelType,t,x,predInterval,seqSteps,tForceStop);
    % test points
    tTestIndices = zeros(1,numTime);
    for k = 1:numTime
        indices = find(t<=refTime(k));
        tTestIndices(1,k) = indices(end);
    end
    rmseErr = root_square_err(tTestIndices,x(:,1:4),xp(:,1:4));
    idx = 4*(i-1);
    errs(idx+1,:) = rmseErr(1,:);
    errs(idx+2,:) = rmseErr(2,:);
    errs(idx+3,:) = rmseErr(3,:);
    errs(idx+4,:) = rmseErr(4,:);
end
disp(["model rmse",mean(errs,1)])
disp(["one rmse",mean(errs,'all')])

disp("plot time step rsme")
figure('Position',[500,100,800,300]); 
tiledlayout("vertical","TileSpacing","tight")
plot(refTime,mean(errs,1),'k-','LineWidth',2);
xlabel("Time (s)","FontName","Arial");
ylabel("Average RMSE","FontName","Arial");
xticks([1,2,3,4,5,6,7,8,9,10]);
% yticks([0,0.2,0.4,0.6,0.8,1])
set(gca, 'FontSize', 15);

%% Prediction Speed Evaluation
tPred = 3;
tSpan = [0,tForceStop+tPred];
% simulation time of ode
tic;
y = sdpm_simulation(tSpan,ctrlOptions);
t_ode = toc;
t = y(:,1);
x = y(:,4:9);
numTime = length(t);
initIdx = find(t >= tForceStop,1,'first');
% predict time of deep learning model
tic
switch modelType
    case "lstm"
        startIdx = initIdx-seqSteps+1;
        x0 = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
    otherwise
        x0 = x(initIdx,:);
end
xp = predict_step_state(net,modelType,x0,tPred);
t_dlm = toc;

disp(["ode",t_ode,"dlm",t_dlm]);

%% supporting functions
function xp = predict_motion(net,type,t,x,predInterval,seqSteps,tForceStop)
    % prediction
    numTime = length(t);
    initIdx = find(t >= tForceStop,1,'first');
    xp = zeros(numTime,6);
    xp(1:initIdx,:) = x(1:initIdx,:);
    switch type
        case "dnn"
            x0 = x(initIdx,:);
            t0 = t(initIdx);
            for i = initIdx+1:numTime
                if (t(i)-t0) > predInterval
                    t0 = t(i-1);
                    x0 = xp(i-1,:);
                end
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
            end
        case "lstm"
            startIdx = initIdx-seqSteps+1;
            x0 = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
            t0 = t(initIdx);
            for i = initIdx+1:numTime          
                if (t(i)-t0) >= predInterval
                    initIdx = i-1;
                    startIdx = initIdx-seqSteps+1;
                    x0 = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
                    t0 = t(initIdx);
                end
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
            end
        case "pinn"
            x0 = x(initIdx,:);
            t0 = t(initIdx);
            for i = initIdx+1:numTime
                if (t(i)-t0 > predInterval)
                    t0 = t(i-1);
                    x0 = xp(i-1,:);
                end
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
            end
        otherwise
            disp("unsupport type model");
    end
end

function xp = predict_step_state(net,type,xInit,tPred)
    xp = zeros(1,6);
    switch type
        case "dnn"
            xp = predict(net,[xInit,tPred]);
        case "lstm"
            dsState = arrayDatastore(xInit,'OutputType',"same",'ReadSize',128);
            dsTime = arrayDatastore(tPred,'ReadSize',128);
            dsTest = combine(dsState, dsTime);
            xp = predict(net,dsTest);
        case "pinn"
            xInit = dlarray([xInit(1:4),tPred]','CB');
            xp(1:4) = extractdata(predict(net,xInit));
        otherwise 
            disp("unsupport model type")
    end
end
