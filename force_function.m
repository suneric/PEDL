function force = force_function(t, fMax, fSpan, fType)
% Create control forces 
    tStart = fSpan(1);
    tEnd = fSpan(2);
    if t <= tEnd && t >= tStart
        switch fType
            case "constant"
                force = fMax;
            case "increase"
                force = fMax*(t-tStart)/(tEnd-tStart); 
            case "decrease"
                force = fMax-fMax*(t-tStart)/(tEnd-tStart);
            otherwise
                force = fMax;
        end
    else
        force = zeros(2,1);
    end
end