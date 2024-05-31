%%
close all;
clear; 
clc;

%% Generate Data for Training 
% Mass-Spring-Damper-Pendulum Dynamics System Parameters
params = parameters();
ctrlOptions = control_options();
strType = {'constant','increase','decrease'};
tSpan = [0,5];
F1Min = max(10,params(10));
F1Range = 10;
% simulate and save data
num_samples = 100;
samples = {};
for i = 1:num_samples
    ctrlOptions.fMax = [F1Min;0]+rand(2,1).*[F1Range;0]; % random max forces
    % ctrlOptions.fType = strType{randi(numel(strType))};
    % ctrlOptions.fSpan = [0,randi([1,5])];
    y = sdpm_simulation(tSpan, ctrlOptions);
    state = y';
    fname=['data/input',num2str(i),'.mat'];
    save(fname, 'state');
    samples{end+1} = fname;
    % plot_states(y(:,1),y(:,4:9),[],'none');
    % plot_forces(y(:,1),y(:,2),y(:,10));
end
samples = reshape(samples,[],1); % make it row-based
save('trainingData.mat','samples');

%% 
function plot_forces(t,f1,fc)
    figure('Position',[500,100,800,800]);
    plot(t,f1,'k-',t,fc,'b-','LineWidth',2);
    legend("F1","Fc");
end