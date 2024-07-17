function dxdt = sdpm_system(t, x, sysParams, ctrlParams)
    F = force_function(t, ctrlParams);
    fc = coulomb_friction(x(2), sysParams, ctrlParams.friction);
    dxdt = compute_xdot(x, F, fc, sysParams);
end