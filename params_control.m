function params = params_control()
    params = struct();
    params.fType = "constant"; % increase, decrease, constant
    params.fSpan = [0, 1]; % applying force for 0 ~ 1 second
    params.fMax = [10; 0]; % maximum [f1;f2], keeping f2 = 0
    params.friction = "andersson"; % none, smooth, andersson, specker
    params.fixedTimeStep = 0; % 0 for varying time step, else for fixed stime step in simulation e.g., 1e-2
    % To many data points will be generated if using default ode options
    % To select small set of data for training with different methods.
    params.method = "interval"; % random, interval, origin
    params.numPoints = 200;
    params.interval = 1e-2;
end