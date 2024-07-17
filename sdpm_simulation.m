function y = sdpm_simulation(tSpan, sysParams, ctrlParams)
    % ODE solver
    % opts = odeset('MaxStep',0.5);
    x0 = zeros(4, 1);
    [t,x] = ode45(@(t,x) sdpm_system(t, x, sysParams, ctrlParams), tSpan, x0);
    % sample time points
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