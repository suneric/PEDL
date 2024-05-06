function [n,states,times,labels] = create_data_next(sample,seq_steps,tForceStop)
    % a sample of sequence data containing time, states, and control inputs
    states = {};    
    times = [];
    labels = [];    
    n = 0;
    t = sample(1,:);
    u = sample(2:3,:);
    x = sample(4:9,:);
    % predict next state with previous sequence steps
    size = length(t);
    indices = find(t <= tForceStop);
    for i = indices(end):size-1
        startIdx = i-seq_steps+1;
        endIdx = i;
        nextIdx = i+1;
        states{end+1} = [t(:,startIdx:endIdx);x(:,startIdx:endIdx);u(:,startIdx:endIdx)];
        times = [times,t(nextIdx)-t(endIdx)];
        labels = [labels,x(:,nextIdx)];
        n = n+1;
    end
end