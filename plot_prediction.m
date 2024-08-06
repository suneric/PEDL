function plot_prediction(modelFile, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, type)
% Single case prediction accuracy over specified time span
    net = load(modelFile).net;
    ctrlParams.fMax = [f1Max; 0]; 
    y = sdpm_simulation(tSpan, sysParams, ctrlParams);
    t = y(:, 1);
    x = y(:, 2:7);
    xp = predict_motion(net, type, t, x, predInterval, trainParams.sequenceStep, ctrlParams.fSpan(2));
    
    initIdx = find(t >= ctrlParams.fSpan(2), 1, 'first');
    tp = t(initIdx:end);
    xp = xp(initIdx:end,:);

    plot_compared_states(t,x,tp,xp);
end

