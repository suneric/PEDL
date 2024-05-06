function dxdt = sdpm_system(t,x,fMax,fSpan,fType)
    F = force_function(t,fMax,fSpan,fType);
    dxdt = compute_xdot(x,F);
end