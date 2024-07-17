function plot_prediction(modelFile,sysParams,ctrlParams,trainParams,fRange,predInterval,tSpan)
% Single case prediction accuracy over specified time span
    net = load(modelFile).net;
    f1Min = max(15, sysParams.fc_max);
    ctrlParams.fMax = [f1Min+fRange; 0]; 
    y = sdpm_simulation(tSpan, sysParams, ctrlParams);
    t = y(:, 1);
    x = y(:, 2:7);
    xp = predict_motion(net, trainParams.type, t, x, predInterval, trainParams.sequenceStep, ctrlParams.fSpan(2));
    
    initIdx = find(t >= ctrlParams.fSpan(2), 1, 'first');
    tp = t(initIdx:end);
    xp = xp(initIdx:end,:);
    plot_compared_states(t,x,tp,xp)
end

