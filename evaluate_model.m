function avgErr = evaluate_model(modelFile, sysParams, ctrlParams, trainParams, tSpan, predInterval, numCase, numTime, type)
    net = load(modelFile).net;
    % evaluate time span, larger time span will increase the simulation
    % time when complicated friction involved
    % test F1 range from 15N ~ 35N
    f1Min = 5; 
    f1Range = linspace(0, 20, numCase);
    
    % reference time points 
    switch trainParams.type
        case {"dnn2", "lstm2", "pinn2"} 
            errs = zeros(2*numCase, numTime);
        case {"dnn4", "lstm4", "pinn4"} 
            errs = zeros(4*numCase, numTime);
        case {"dnn6", "lstm6", "pinn6"} 
            errs = zeros(6*numCase, numTime);
        otherwise
            disp("unspecify type of model.")
    end
    for i = 1:numCase
        f1Max = f1Min+f1Range(i); 
        ctrlParams.fMax = [f1Max; 0];
        y = sdpm_simulation(tSpan, sysParams, ctrlParams);
        t = y(:,1);
        x = y(:,2:7);
        [xp, rmseErr, refTime] = evaluate_single(net, t, x, ctrlParams, trainParams, tSpan, predInterval, numTime, type);
        disp("evaluate "+num2str(i)+" th case, f1: "+num2str(f1Max) + " N, mean square err: " + num2str(mean(rmseErr, "all")));
        switch trainParams.type
            case {"dnn2", "lstm2", "pinn2"} 
                errs(2*(i-1)+1:2*(i-1)+2,:) = rmseErr;
            case {"dnn4", "lstm4", "pinn4"} 
                errs(4*(i-1)+1:4*(i-1)+4,:) = rmseErr;
            case {"dnn6", "lstm6", "pinn6"} 
                errs(6*(i-1)+1:6*(i-1)+6,:) = rmseErr;    
            otherwise
                disp("unspecify type of model.")
        end
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



