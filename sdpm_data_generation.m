%%
close all;
clear; 
clc;

%% Generate Data for Training 
% Mass-Spring-Damper-Pendulum Dynamics System Parameters
tSpan = [0,5];
ctrlOptions = control_options();
strType = {'constant','increase','decrease'};

% simulate and save data
num_samples = 600;
samples = {};
for i = 1:num_samples
    ctrlOptions.fMax = rand(2,1).*[10;0]; % random max forces
    % ctrlOptions.fType = strType{randi(numel(strType))};
    % ctrlOptions.fSpan = [0,randi([1,5])];
    y = sdpm_simulation(tSpan, ctrlOptions);
    state = y';
    fname=['data/input',num2str(i),'.mat'];
    save(fname, 'state');
    samples{end+1} = fname;
    % plot_states(y(:,1),y(:,4:9),[],'none')
end
samples = reshape(samples,[],1); % make it row-based
save('trainingData.mat','samples');