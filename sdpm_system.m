function dxdt = sdpm_system(t,x,fMax,fSpan,fType)
    F = force_function(t,fMax,fSpan,fType);
    fc = coulomb_friction(x(2),F(1));
    dxdt = compute_xdot(x,F,fc);
end