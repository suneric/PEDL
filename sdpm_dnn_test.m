%%
close all;
clear;
clc;

%% set task type
% lossType = 'PgNN';
% lossType = 'PiNN';
lossType = 'PcNN'; % combine data loss and physics loss
task = "predict_next";
% task = "predict_arbitrary";
seq_steps = 20;
t_force_stop = 1;

num_samples = 300;
fname = "model/"+lossType+"_model_"+num2str(num_samples)+".mat";
model = load(fname).net;
% plot(model)
% Mass-Spring-Damper-Pendulum Dynamics System Parameters
tSpan = [0,10];
ctrlOptions = control_options();
strType = {'constant','increase','decrease'};

%% Test
max_forces = linspace(0.5,15,30);
tTest = linspace(1,10,50);
num_test = length(max_forces);
err_list = zeros(6*num_test,length(tTest));
for i = 1:num_test
    disp("test case: "+num2str(i)+" / "+num2str(num_test));
    ctrlOptions.fMax = [max_forces(i);0];
    % ctrlOptions.fMax = rand(2,1).*[10;0]; % random max forces
    %ctrlOptions.fType = strType{randi(numel(strType))};
    % ctrlOptions.fSpan = [0,randi([2,5])];
    y = sdpm_simulation(tSpan,ctrlOptions);
    t = y(:,1);
    u = y(:,2:3);
    x = y(:,4:9);
    x_pred = predict_motion(model,t,x,u,seq_steps,tForceStop,task);
    % plot_states(t,x,ctrlOptions,x_pred)
    startRow = 6*(i-1)+1;
    err_list(startRow:startRow+5,:) = rmse(t,x,x_pred,tTest);
end
save(['test/',lossType,'_',num2str(num_samples),'.mat'], 'err_list');

function err = rmse(t,x,xPred,tTest)
    numPoints = length(tTest);
    errs = zeros(6,numPoints);
    for i = 1:numPoints
        indices = find(t < tTest(i));
        errs(1,i) = x(indices(end)+1,1)-xPred(indices(end)+1,1);
        errs(2,i) = x(indices(end)+1,2)-xPred(indices(end)+1,2);
        errs(3,i) = x(indices(end)+1,3)-xPred(indices(end)+1,3);
        errs(4,i) = x(indices(end)+1,4)-xPred(indices(end)+1,4);
        errs(5,i) = x(indices(end)+1,5)-xPred(indices(end)+1,5);
        errs(6,i) = x(indices(end)+1,6)-xPred(indices(end)+1,6);
    end
    err = sqrt(errs.^2);
end
