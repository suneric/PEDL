function avgErr = evaluate_model(modelFile, sysParams, ctrlParams, trainParams)
    net = load(modelFile).net;
    % evaluate time span, larger time span will increase the simulation
    % time when complicated friction involved
    tSpan = [0,7]; 
    predInterval = tSpan(2); % predict maximum time interval
    % test F1 range from 15N ~ 35N
    numCase = 30; % evaluate cases
    f1Min = 15; 
    f1Range = linspace(0, 20, numCase);
    % reference time points
    numTime = 50; % evaluate time points 
    refTime = linspace(1, tSpan(2), numTime); 
    errs = zeros(6*numCase, numTime);
    for i = 1:numCase
        disp("evaluate "+num2str(i)+" th case.");
        ctrlParams.fMax = [f1Min+f1Range(i); 0]; 
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
    xticks(linspace(1,tSpan(2),(tSpan(2))));
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

