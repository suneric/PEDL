function xp = predict_motion(net, type, t, x, predInterval, seqSteps, tForceStop)
    numTime = length(t);
    initIdx = find(t >= tForceStop, 1, 'first');
    xp = zeros(numTime, 6);
    xp(1:initIdx, :) = x(1:initIdx, :);
    switch type
        case "dnn"
            x0 = x(initIdx, :);
            t0 = t(initIdx);
            for i = initIdx+1 : numTime
                if (t(i)-t0) > predInterval
                    t0 = t(i-1);
                    x0 = xp(i-1, :);
                end
                xp(i,:) = predict_step_state(net, type, x0, t(i)-t0);
            end
        case "lstm"
            startIdx = initIdx-seqSteps+1;
            x0 = {[t(startIdx:initIdx), xp(startIdx:initIdx,:)]'};
            t0 = t(initIdx);
            for i = initIdx+1 : numTime          
                if (t(i)-t0) >= predInterval
                    initIdx = i-1;
                    startIdx = initIdx-seqSteps+1;
                    x0 = {[t(startIdx:initIdx), xp(startIdx:initIdx,:)]'};
                    t0 = t(initIdx);
                end
                xp(i,:) = predict_step_state(net, type, x0, t(i)-t0);
            end
        case "pinn"
            x0 = x(initIdx, :);
            t0 = t(initIdx);
            for i = initIdx+1 : numTime
                if (t(i)-t0 > predInterval)
                    t0 = t(i-1);
                    x0 = xp(i-1, :);
                end
                xp(i,:) = predict_step_state(net, type, x0, t(i)-t0);
            end
        otherwise
            disp("unspecified type of model");
    end
end

function xp = predict_step_state(net, type, xInit, tPred)
    xp = zeros(1,6);
    switch type
        case "dnn"
            xp = predict(net, [xInit, tPred]);
        case "lstm"
            dsState = arrayDatastore(xInit, 'OutputType', 'same', 'ReadSize',1);
            dsTime = arrayDatastore(tPred, 'ReadSize', 1);
            dsTest = combine(dsState, dsTime);
            xp = predict(net, dsTest);
        case "pinn"
            xInit = dlarray([xInit, tPred]', 'CB');
            xp = extractdata(predict(net, xInit));
        otherwise 
            disp("unspecified type of model")
    end
end