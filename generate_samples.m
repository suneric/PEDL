function dataFile = generate_samples(sysParams, ctrlParams, trainParams)
% Generate samples and save the data file into a subfolder "/data"
    dataFile = './trainingSamples.mat';
    % check whether need to regenerate samples
    regenerate_samples = 1; % by default, regrenerate samples
    if exist(dataFile, 'file') == 2
        ds = load(dataFile);
        if trainParams.numSamples == length(ds.samples)
            regenerate_samples = 0;
        end
    end
    
    % generate sample data
    if regenerate_samples      
        samples = {};
        f1Min = max(15, sysParams.fc_max);
        for i = 1:trainParams.numSamples
            disp(["generate data for", num2str(i), "th sample."]);
            % random max force F1 for each sample in a varying range of 10N
            ctrlParams.fMax = [f1Min; 0]+rand(2,1).*[10; 0]; 
            y = sdpm_simulation([0,5], sysParams, ctrlParams);
            state = y';
            fname=['./data/input',num2str(i),'.mat'];
            save(fname, 'state');
            samples{end+1} = fname;
        end
        samples = reshape(samples, [], 1); % make it row-based
        save(dataFile, 'samples');
    else
        disp(num2str(trainParams.numSamples) + " samples is already generated.");
    end
end