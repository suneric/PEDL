function [n,states,times,labels] = create_data_arbitrary(sample,num_steps,tForceStop)
    % a sample of sequence data containing time, states, and control inputs
    states = {};    
    times = [];
    labels = [];    
    n = 0; % number of data points generated
    t = sample(1,:);
    u = sample(2:3,:);
    x = sample(4:9,:);
    % predict arbitary time step with the first 1-s input
    size = length(t);
    indices = find(t <= tForceStop);
    initIdx = indices(end);
    startIdx = initIdx-num_steps+1;
    t0 = t(initIdx);
    x0 = [t(1,startIdx:initIdx);x(:,startIdx:initIdx);u(:,startIdx:initIdx)];
    for i=initIdx+1:size
        states{end+1} = x0;
        times = [times,t(i)-t0];
        labels = [labels,x(:,i)];
        n = n+1;
    end
end