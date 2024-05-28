function x_pred = predict_motion(model,t,x,u,seq_steps,tForceStop,task)
    switch task
        case "predict_next"
            x_pred = predict_next_step(model,t,x,u,seq_steps,tForceStop);
        case "predict_arbitrary"
            x_pred = predict_arbitary_step(model,t,x,u,seq_steps,tForceStop);
        otherwise
            disp("unspecified task")
    end
end

function x_pred = predict_next_step(model,t,x,u,seq_steps,tForceStop)
    size = length(t);
    x_pred = zeros(size,6);
    indices = find(t <= tForceStop);
    x_pred(1:indices(end),:) = x(1:indices(end),:);
    for i = indices(end):size-1
        startIdx = i-seq_steps+1;
        endIdx = i;
        nextIdx = i+1;
        state = {[t(startIdx:endIdx),x_pred(startIdx:endIdx,:),u(startIdx:endIdx,:)]'};
        dsState = arrayDatastore(state,'OutputType',"same",'ReadSize',128);
        dTime = t(i+1)-t(i);
        dsTime = arrayDatastore(dTime,'ReadSize',128);
        dsTest = combine(dsState, dsTime);
        x_pred(nextIdx,:) = predict(model,dsTest);
    end
end

function x_pred = predict_arbitary_step(model,t,x,u,num_steps,tForceStop)
    size = length(t);
    x_pred = zeros(size,6);
    indices = find(t <= tForceStop);
    x_pred(indices,:) = x(indices,:);
    initIdx = indices(end);
    startIdx = initIdx-num_steps+1;
    state = {[t(startIdx:initIdx),x_pred(startIdx:initIdx,:),u(startIdx:initIdx,:)]'};
    dsState = arrayDatastore(state,'OutputType',"same",'ReadSize',128);
    t0 = t(initIdx);
    for i = initIdx+1:size    
        dsTime = arrayDatastore(t(i)-t0,'ReadSize',128);
        dsTest = combine(dsState, dsTime);
        x_pred(i,:) = predict(model,dsTest);
    end
end