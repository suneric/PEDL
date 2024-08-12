function [dataFile, fMaxRange] = generate_samples(sysParams, ctrlParams, trainParams, f1Max, tSpan)
% Generate samples and save the data file into a subfolder "data\"
    dataFile = "trainingSamples.mat";
    fMaxRange = zeros(trainParams.numSamples,1);
    % check whether need to regenerate samples
    regenerate_samples = 1; % by default, regrenerate samples
    if exist(dataFile, 'file') == 2
        ds = load(dataFile);
        if trainParams.numSamples == length(ds.samples)
            regenerate_samples = 0;
            for i = 1:length(ds.samples)
                data = load(ds.samples{i,1}).state;
                fMaxRange(i) = data(8,1); % f1
            end
        end
    end
    
    % generate sample data
    if regenerate_samples      
        samples = {};
        f1Min = f1Max(1);
        f1Range = f1Max(2)-f1Max(1);
        for i = 1:trainParams.numSamples
            disp("generate data for " + num2str(i) + "th sample.");
            % random max force F1 for each sample in a varying range of 10N
            ctrlParams.fMax = [f1Min; 0]+rand(2,1).*[f1Range; 0];
            fMaxRange(i) =  ctrlParams.fMax(1); % f1
            y = sdpm_simulation(tSpan, sysParams, ctrlParams);
            state = y';
            fname=['data\input',num2str(i),'.mat'];
            save(fname, 'state');
            samples{end+1} = fname;
        end
        samples = reshape(samples, [], 1); % make it row-based
        save(dataFile, 'samples');
    else
        disp(num2str(trainParams.numSamples) + " samples is already generated.");
    end
end