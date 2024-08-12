function y = sdpm_simulation(tSpan, sysParams, ctrlParams)
    % ODE solver
    if ctrlParams.fixedTimeStep ~= 0
        tSpan = tSpan(1):ctrlParams.fixedTimeStep:tSpan(2);
    end
    x0 = zeros(4, 1); % q1, q1d, q2, q2d
    [t,x] = ode45(@(t,x) sdpm_system(t, x, sysParams, ctrlParams), tSpan, x0);
    % sample time points
    if ctrlParams.fixedTimeStep == 0
        [t,x] = get_samples(ctrlParams, t, x, ctrlParams.tolerance);
    end
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

function [ts, xs] = get_samples(ctrlParams, t, x, tStep)
    if ctrlParams.friction == "andersson"
        ts = [t(1)];
        xs = [x(1,:)];
        for i = 1:length(t)
            if(t(i)-ts(end) >= tStep)
                ts = [ts;t(i)];
                xs = [xs;x(i,:)];
            end
        end
    else
        ts = t;
        xs = x;
    end
end