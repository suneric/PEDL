function F = physics_law(X,Xd,Xdd)
    q1 = X(1,:);
    q2 = X(2,:);
    q1d = Xd(1,:);
    q2d = Xd(2,:);
    q1dd = Xdd(1,:);
    q2dd = Xdd(2,:);
    % system parameters
    sysParams = params_system();
    K = sysParams.K;
    C = sysParams.C;
    L = sysParams.L;
    G = sysParams.G;
    M1 = sysParams.M1;
    M2 = sysParams.M2;
    ctrlParams = params_control();
    % columb friction
    fc = coulomb_friction(q1d, sysParams, ctrlParams.friction);
    % Lagrangian equation: F - fc = M*q_ddot + V*q_dot + G
    f1 = (M1+M2)*q1dd + M2*L*(cos(q2).*q2dd) + C*q1d - M2*L*(sin(q2).*q2d.^2) + K*q1 + fc;
    f2 = M2*L*(cos(q2).*q1dd) + M2*L*L*q2dd + M2*G*L*sin(q2);
    F = [f1;f2];
end