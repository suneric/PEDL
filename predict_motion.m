function xp = predict_motion(net, type, t, x, predInterval, seqSteps, tForceStop)
    numTime = length(t);
    initIdx = find(t > tForceStop, 1, 'first'); % start where force stop acting
    switch type
        case "dnn6"
            xp = zeros(numTime, 6);
            xp(1:initIdx, :) = x(1:initIdx, :);
            x0 = x(initIdx, :);
            t0 = t(initIdx);
            for i = initIdx+1 : numTime
                if (t(i)-t0) > predInterval
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    x0 = x(idx, :);
                    t0 = t(idx);
                end
                xp(i,:) = predict(net, [x0, t(i)-t0]);
            end
        case "lstm6"
            xp = zeros(numTime, 6);
            xp(1:initIdx, :) = x(1:initIdx, :);
            startIdx = initIdx-seqSteps+1;
            x0 = {[t(startIdx:initIdx), x(startIdx:initIdx,:)]'};
            t0 = t(initIdx);
            for i = initIdx+1 : numTime          
                if (t(i)-t0) > predInterval
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    startIdx = idx-seqSteps+1;
                    x0 = {[t(startIdx:idx), x(startIdx:idx,:)]'};
                    t0 = t(idx);
                end
                dsState = arrayDatastore(x0, 'OutputType', 'same', 'ReadSize',1);
                dsTime = arrayDatastore(t(i)-t0, 'ReadSize', 1);
                dsTest = combine(dsState, dsTime);
                xp(i,:) = predict(net, dsTest);
            end
        case {"pinn6", "pirn6"}
            xp = zeros(numTime, 6);
            xp(1:initIdx, :) = x(1:initIdx, :);
            x0 = x(initIdx, :);
            t0 = t(initIdx);
            for i = initIdx+1 : numTime
                if (t(i)-t0 > predInterval)
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    x0 = x(idx, :);
                    t0 = t(idx);
                end
                xp(i,:) = extractdata(predict(net, dlarray([x0, t(i)-t0]', 'CB')));
            end
        case "dnn4"
            xp = zeros(numTime, 4);
            xp(1:initIdx, :) = x(1:initIdx, 1:4);
            x0 = x(initIdx, 1:4);
            t0 = t(initIdx);
            for i = initIdx+1 : numTime
                if (t(i)-t0) > predInterval
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    x0 = x(idx, 1:4);
                    t0 = t(idx);                    
                end
                xp(i,:) = predict(net, [x0, t(i)-t0]);
            end
        case "lstm4"
            xp = zeros(numTime, 4);
            xp(1:initIdx, :) = x(1:initIdx, 1:4);
            startIdx = initIdx-seqSteps+1;
            x0 = {[t(startIdx:initIdx), x(startIdx:initIdx,1:4)]'};
            t0 = t(initIdx);
            for i = initIdx+1 : numTime          
                if (t(i)-t0) > predInterval
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    startIdx = idx-seqSteps+1;
                    x0 = {[t(startIdx:idx), x(startIdx:idx,1:4)]'};
                    t0 = t(idx);
                end
                dsState = arrayDatastore(x0, 'OutputType', 'same', 'ReadSize',1);
                dsTime = arrayDatastore(t(i)-t0, 'ReadSize', 1);
                dsTest = combine(dsState, dsTime);
                xp(i,:) = predict(net, dsTest);
            end
        case {"pinn4", "pirn4"}
            xp = zeros(numTime, 4);
            xp(1:initIdx, :) = x(1:initIdx, 1:4);
            x0 = x(initIdx, 1:4);
            t0 = t(initIdx);
            for i = initIdx+1 : numTime
                if (t(i)-t0 > predInterval)
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    x0 = x(idx, 1:4);
                    t0 = t(idx);
                    
                end
                xp(i,:) = extractdata(predict(net, dlarray([x0, t(i)-t0]', 'CB')));
            end
        case "dnn2"
            xp = zeros(numTime, 2);
            xp(1:initIdx, :) = x(1:initIdx, 1:2);
            x0 = x(initIdx, 1:2);
            t0 = t(initIdx);
            for i = initIdx+1 : numTime
                if (t(i)-t0) > predInterval
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    x0 = x(idx, 1:2);
                    t0 = t(idx);
                end
                xp(i,:) = predict(net, [x0, t(i)-t0]);
            end
        case "lstm2"
            xp = zeros(numTime, 2);
            xp(1:initIdx, :) = x(1:initIdx, 1:2);
            startIdx = initIdx-seqSteps+1;
            x0 = {[t(startIdx:initIdx), x(startIdx:initIdx,1:2)]'};
            t0 = t(initIdx);
            for i = initIdx+1 : numTime          
                if (t(i)-t0) > predInterval
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    startIdx = idx-seqSteps+1;
                    x0 = {[t(startIdx:idx), x(startIdx:idx,1:2)]'};
                    t0 = t(idx);
                end
                dsState = arrayDatastore(x0, 'OutputType', 'same', 'ReadSize',1);
                dsTime = arrayDatastore(t(i)-t0, 'ReadSize', 1);
                dsTest = combine(dsState, dsTime);
                xp(i,:) = predict(net, dsTest);
            end
        case {"pinn2", "pirn2"}
            xp = zeros(numTime, 2);
            xp(1:initIdx, :) = x(1:initIdx, 1:2);
            x0 = x(initIdx, 1:2);
            t0 = t(initIdx);
            for i = initIdx+1 : numTime
                if (t(i)-t0 > predInterval)
                    idx = find(t >= t0+floor(predInterval/2), 1, 'first');
                    x0 = x(idx, 1:2);
                    t0 = t(idx);
                end
                xp(i,:) = extractdata(predict(net, dlarray([x0, t(i)-t0]', 'CB')));
            end
        otherwise
            disp("unspecified type of model");
    end
end