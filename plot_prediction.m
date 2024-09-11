function [t,x,xp] = plot_prediction(net, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, type, numState)
% Single case prediction accuracy over specified time span
    ctrlParams.fMax = [f1Max; 0]; 
    y = sdpm_simulation(tSpan, sysParams, ctrlParams);
    t = y(:, 1);
    x = y(:, 2:7);
    xp = predict_motion(net, type, t, x, predInterval, trainParams.sequenceStep, ctrlParams.fSpan(2));
    
    initIdx = find(t >= tSpan(1), 1, 'first');
    ptp = t(initIdx:end)-1;
    pxp = xp(initIdx:end,:);
    px = x(initIdx:end,:);
    plot_compared_states_t(ptp,px,ptp,pxp,numState);
    % plot_compared_states_pv(px,pxp);
end

