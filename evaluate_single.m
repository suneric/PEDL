function [xp, rmseErr, refTime] = evaluate_single(net, t, x, ctrlParams, trainParams, tSpan, predInterval, numTime, type)
    % predict state use the trained model
    xp = predict_motion(net, type, t, x, predInterval, trainParams.sequenceStep, ctrlParams.fSpan(2));

    % test reference points
    tTestIndices = zeros(numTime,1);
    refTime = linspace(1, tSpan(2), numTime); 
    for k = 1:numTime
        indices = find(t <= refTime(k), 1, 'last');
        tTestIndices(k) = indices(end);
    end
    rmseErr = root_square_err(tTestIndices, x, xp);
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