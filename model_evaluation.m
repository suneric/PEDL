%%
close all;
clear;
clc;
disp("clear and start the program")

%% set task type
seqSteps = 20;
tForceStop = 1;% time stop force
tSpan = [0,10]; % simulation time span
ctrlOptions = control_options();
disp("initialize parameters");

modelType = "pinn"; % "dnn", "pinn", "lstm"
numSamples = 200;
modelFile = "model/"+modelType+"_"+num2str(ctrlOptions.alpha)+"_"+num2str(numSamples)+".mat";
net = load(modelFile).net;
predictTime = 3;

%% Single case prediction accuracy over specified time span
ctrlOptions.fMax = [8;0];
y = sdpm_simulation(tSpan, ctrlOptions);
t = y(:,1);
x = y(:,4:9);
xp = predict_motion(net,modelType,t,x,predictTime,seqSteps,tForceStop);

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
numCase = 30;
numTime = 30;
refTime = linspace(1,10,numTime);
maxForces = linspace(0.5,15,numCase);
errs = zeros(4*numCase,numTime);
for i = 1:numCase
    % reference
    ctrlOptions.fMax = [maxForces(i);0];
    y = sdpm_simulation(tSpan, ctrlOptions);
    t = y(:,1);
    x = y(:,4:9);
    xp = predict_motion(net,modelType,t,x,predictTime,seqSteps,tForceStop);
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

disp("plot time step rsme")
figure('Position',[500,100,800,300]); 
tiledlayout("vertical","TileSpacing","tight")
plot(refTime,mean(errs,1),'k-','LineWidth',2);
xlabel("Time (s)","FontName","Arial");
ylabel("Average RMSE","FontName","Arial");
xticks([1,2,3,4,5,6,7,8,9,10]);
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
indices = find(t <= tForceStop);
initIdx = indices(end);
% predict time of deep learning model
tic
xp = predict_step_state(net,modelType,x(initIdx,:),tPred);
t_dlm = toc;

disp(["ode",t_ode,"dlm",t_dlm]);

%% supporting functions
function xp = predict_motion(net,type,t,x,predictTime,seqSteps,tForceStop)
    % prediction
    numTime = length(t);
    indices = find(t <= tForceStop);
    initIdx = indices(end);
    xp = zeros(numTime,6);
    xp(indices,:) = x(indices,:);
    switch type
        case "dnn"
            x0 = x(initIdx,:);
            t0 = t(initIdx);
            for i = initIdx+1:numTime
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
                if (t(i)-t0) > predictTime
                    t0 = t(i-1);
                    x0 = xp(i-1,:);
                end
            end
        case "lstm"
            startIdx = initIdx-seqSteps+1;
            x0 = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
            t0 = t(initIdx);
            for i = initIdx+1:numTime
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
                if (t(i)-t0) >= predictTime
                    initIdx = i-1;
                    startIdx = initIdx-seqSteps+1;
                    x0 = {[t(startIdx:initIdx),xp(startIdx:initIdx,:)]'};
                    t0 = t(initIdx);
                end
            end
        case "pinn"
            x0 = x(initIdx,:);
            t0 = t(initIdx);
            for i = initIdx+1:numTime
                xp(i,:) = predict_step_state(net,type,x0,t(i)-t0);
                if (t(i)-t0 > predictTime)
                    t0 = t(i-1);
                    x0 = xp(i-1,:);
                end
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
            xInit = dlarray([xInit(1,1:4),tPred]','CB');
            xPred = predict(net,xInit);
            xp(1,1:4) = extractdata(xPred);
        otherwise 
            disp("unsupport model type")
    end
end
