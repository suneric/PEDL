function avgErr = evaluate_model(modelFile, sysParams, ctrlParams, trainParams)
    net = load(modelFile).net;
    numCase = 30; % evaluate cases
    numTime = 60; % evaluate time points 
    tSpan = [0,10]; % evaluate time span
    predInterval = 10; % predict maximum time interval
    
    errs = zeros(6*numCase, numTime);

    fRanges = linspace(0.5, 15, numCase);
    refTime = linspace(1, 10, numTime); % reference time points
    for i = 1:numCase
        disp("evaluate "+num2str(i)+" th case.");
        ctrlParams.fMax = [sysParams.fc_max+fRanges(i); 0]; 
        y = sdpm_simulation(tSpan, sysParams, ctrlParams);
        t = y(:,1);
        x = y(:,2:7);
        xp = predict_motion(net, trainParams.type, t, x, predInterval, trainParams.sequenceStep, ctrlParams.fSpan(2));
        
        % test reference points
        tTestIndices = zeros(1, numTime);
        for k = 1:numTime
            indices = find(t <= refTime(k));
            tTestIndices(1,k) = indices(end);
        end
        rmseErr = root_square_err(tTestIndices, x, xp);
        idx = 6*(i-1); 
        errs(idx+1,:) = rmseErr(1,:);
        errs(idx+2,:) = rmseErr(2,:);
        errs(idx+3,:) = rmseErr(3,:);
        errs(idx+4,:) = rmseErr(4,:);
        errs(idx+5,:) = rmseErr(5,:);
        errs(idx+6,:) = rmseErr(6,:);
    end
    avgErr = mean(errs,'all');
   
    disp("plot time step rsme")
    figure('Position',[500,100,800,300]); 
    tiledlayout("vertical","TileSpacing","tight")
    plot(refTime,mean(errs,1),'k-','LineWidth',2);
    xlabel("Time (s)","FontName","Arial");
    ylabel("Average RMSE","FontName","Arial");
    xticks([1,2,3,4,5,6,7,8,9,10]);
    set(gca, 'FontSize', 15);
end

% root square error of prediction and reference
function rse = root_square_err(indices, x, xp)
    numPoints = length(indices);
    x_size = size(xp);
    errs = zeros(x_size(2), numPoints);
    for i = 1 : numPoints
        for j = 1:x_size(2)
            errs(j, i) = x(indices(i), j) - xp(indices(i), j);
        end
    end
    rse = sqrt(errs.^2);
end

