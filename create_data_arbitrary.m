function [n,states,times,labels] = create_data_arbitrary(sample,num_steps,tForceStop)
    % a sample of sequence data containing time, states, and control inputs
    states = {};    
    times = [];
    labels = [];    
    n = 0;
    t = sample(1,:);
    u = sample(2:3,:);
    x = sample(4:9,:);
    % predict arbitary time step with the first 1-s input
    indices = find(t <= tForceStop);
    for i=indices(end)+1:length(t)
        randomIndices = sort(randperm(numel(indices),num_steps));
        state = [t(1,randomIndices);x(:,randomIndices);u(:,randomIndices)];
        states{end+1} = state;
        times = [times,t(i)];
        labels = [labels,x(:,i)];
        n = n+1;
    end
end