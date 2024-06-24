function y = sdpm_simulation(tSpan, ctrlOptions)
    % ODE solver
    x0 = zeros(4,1);
    fMax = ctrlOptions.fMax;
    fSpan = ctrlOptions.fSpan;
    fType = ctrlOptions.fType;
    % opts = odeset('MaxStep',0.5);
    [t,x] = ode45(@(t,x) sdpm_system(t,x,fMax,fSpan,fType),tSpan,x0);
    % sample time points
    ts = get_sample_times(ctrlOptions,t); 
    size = length(ts);
    y = zeros(size,10); 
    for i = 1:size
        indices = find(t <= ts(i));
        idx = indices(end);
        F = force_function(t(idx),fMax,fSpan,fType);
        fc = coulomb_friction(x(idx,2),F(1));
        xdot = compute_xdot(x(idx,:),F,fc);
        y(i,1) = t(idx); % t
        y(i,2) = F(1); % f1
        y(i,3) = F(2); % f2
        y(i,4) = x(idx,1); % q1
        y(i,5) = x(idx,3); % q2
        y(i,6) = x(idx,2); % q1_dot
        y(i,7) = x(idx,4); % q2_dot
        y(i,8) = xdot(2); % q1_ddot
        y(i,9) = xdot(4); % q2_ddot
        y(i,10) = -fc; % coulomb friction
    end
end

function ts = get_sample_times(ctrlOptions, t)
    if ctrlOptions.friction == 'coulomb'
        ts = [t(1)];
        for i = 1:length(t)
            if(t(i)-ts(end) > ctrlOptions.tSample)
                ts = [ts;t(i)];
            end
        end
    else
        ts = t;
    end
end