function errs = evaluate_model_with_4_states(net, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, numTime, type)
    numCase = length(f1Max);
    errs = zeros(4*numCase, numTime);
    for i = 1:numCase 
        ctrlParams.fMax = [f1Max(i); 0];
        y = sdpm_simulation(tSpan, sysParams, ctrlParams);
        t = y(:,1);
        x = y(:,2:7);
        [xp, rmseErr, refTime] = evaluate_single(net, t, x, ctrlParams, trainParams, tSpan, predInterval, numTime, type);
        allErr = rmseErr(1:4,:);
        disp("evaluate "+num2str(i)+" th case, f1: "+num2str(f1Max(i)) + " N, mean square err: " + num2str(mean(allErr, "all")));
        errs(4*(i-1)+1:4*(i-1)+4,:) = allErr;
    end
end



