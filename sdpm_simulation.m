function y = sdpm_simulation(tSpan, sysParams, ctrlParams)
    % ODE solver
    if ctrlParams.fixedTimeStep ~= 0
        tSpan = tSpan(1):ctrlParams.fixedTimeStep:tSpan(2);
    end
    x0 = zeros(4, 1); % q1, q1d, q2, q2d
    [t,x] = ode45(@(t,x) sdpm_system(t, x, sysParams, ctrlParams), tSpan, x0);
    % sample time points
    [t,x] = get_samples(ctrlParams, t, x);
    numTime = length(t);
    y = zeros(numTime, 10); 
    for i = 1 : numTime
        F = force_function(t(i), ctrlParams);
        fc = coulomb_friction(x(i,2), sysParams, ctrlParams.friction);
        xdot = compute_xdot(x(i,:), F, fc, sysParams);
        y(i,1) = t(i); % t
        y(i,2) = x(i, 1); % q1
        y(i,3) = x(i, 3); % q2
        y(i,4) = x(i, 2); % q1dot
        y(i,5) = x(i, 4); % q2dot
        y(i,6) = xdot(2); % q1ddot
        y(i,7) = xdot(4); % q2ddot
        y(i,8) = F(1); % f1
        y(i,9) = F(2); % f2
        y(i,10) = -fc; % coulomb friction
    end
end

function [ts, xs] = get_samples(ctrlParams, t, x)
    switch ctrlParams.friction
        case {"none","smooth"}
            ts = t;
            xs = x;
        case {"andersson", "specker"}
            [ts, xs] = select_samples(ctrlParams, t, x);
        otherwise
            ts = t;
            xs = x;
    end
end

function [ts, xs] = select_samples(ctrlParams, t, x)
    switch ctrlParams.method
        case "random"
            indices = randperm(length(t), ctrlParams.numPoints);
            sortIndices = sort(indices);
            ts = t(sortIndices);
            xs = x(sortIndices,:);
        case "interval"
            ts = [t(1)];
            xs = [x(1,:)];
            for i = 2:length(t)
                if t(i)-ts(end) >= ctrlParams.interval
                    ts = [ts;t(i)];
                    xs = [xs;x(i,:)];
                end
            end
        otherwise
            ts = t;
            xs = x;
    end
end