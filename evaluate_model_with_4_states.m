function avgErr = evaluate_model_with_4_states(modelFile, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, numCase, numTime, type)
    net = load(modelFile).net;
    % evaluate time span, larger time span will increase the simulation
    % time when complicated friction involved
    % test F1 range from 10N ~ 30N
    f1Min = f1Max(1); 
    f1Range = linspace(0, f1Max(2)-f1Max(1), numCase);

    errs = zeros(4*numCase, numTime);

    for i = 1:numCase
        f1Max = f1Min+f1Range(i); 
        ctrlParams.fMax = [f1Max; 0];
        y = sdpm_simulation(tSpan, sysParams, ctrlParams);
        t = y(:,1);
        x = y(:,2:7);
        [xp, rmseErr, refTime] = evaluate_single(net, t, x, ctrlParams, trainParams, tSpan, predInterval, numTime, type);
        allErr = rmseErr(1:4,:);
        disp("evaluate "+num2str(i)+" th case, f1: "+num2str(f1Max) + " N, mean square err: " + num2str(mean(allErr, "all")));
        errs(4*(i-1)+1:4*(i-1)+4,:) = allErr;
    end
    
    avgErr = mean(errs,'all'); % one value of error for estimtation

    disp("plot time step rsme")
    figure('Position',[500,100,800,300]); 
    tiledlayout("vertical","TileSpacing","tight")
    plot(refTime,mean(errs,1),'k-','LineWidth',2);
    xlabel("Time (s)","FontName","Arial");
    ylabel("Average RMSE","FontName","Arial");
    xticks(linspace(1,tSpan(2),(tSpan(2))));
    title("Average RMSE: "+num2str(avgErr));
    set(gca, 'FontSize', 15);
end



