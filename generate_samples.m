function dataFile = generate_samples(sysParams, ctrlParams, trainParams, f1Max, tSpan)
% Generate samples and save the data file into a subfolder "data\"
    dataFile = "trainingSamples.mat";
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
        for i = 1:length(f1Max)
            disp("generate data for " + num2str(i) + "th sample.");
            % random max force F1 for each sample in a varying range of 10N
            ctrlParams.fMax = [f1Max(i); 0];
            y = sdpm_simulation(tSpan, sysParams, ctrlParams);
            
            numTime = length(y(:,1));
            errLevels = 1 + ctrlParams.noiseLevel*(rand(numTime,6)-0.5)/100;
            states = y(:,2:7);
            states = states.*errLevels;
            % compare_data_error(y(:,1),y(:,2:7),states);
            y(:,2:7) = states;
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

function compare_data_error(t,a,b)
    figure('Position',[500,100,800,500],'Color','White');
    tiledlayout("vertical","TileSpacing","tight")
    for i = 1:6
        nexttile
        plot(t,a(:,i),'k',t,b(:,i),'r');
    end
end