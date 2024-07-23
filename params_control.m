function params = params_control()
    params = struct();
    params.fType = "constant"; % increase, decrease, constant
    params.fSpan = [0, 1]; % applying force for 0 ~ 1 second
    params.fMax = [10; 0]; % maximum [f1;f2], keeping f2 = 0
    params.friction = "andersson"; % none, smooth, andersson
end
