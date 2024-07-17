function force = force_function(t, ctrlParams)
    % apply force (fMax) in range of fSpan according to the fType
    fSpan = ctrlParams.fSpan;
    fMax = ctrlParams.fMax;
    fType = ctrlParams.fType;
    if t <= fSpan(2) && t >= fSpan(1)
        switch fType
            case "constant"
                force = fMax;
            case "increase"
                force = fMax*(t-fSpan(1)) / (fSpan(2)-fSpan(1)); 
            case "decrease"
                force = fMax-fMax*(t-fSpan(1)) / (fSpan(2)-fSpan(1));
            otherwise
                force = fMax;
        end
    else
        force = zeros(2,1);
    end
end