function xdot = compute_xdot(x, F, fc, sysParams)
    q1 = x(1);
    q1d = x(2);
    q2 = x(3);
    q2d = x(4);
    
    % system parameters
    K = sysParams.K;
    C = sysParams.C;
    L = sysParams.L;
    G = sysParams.G;
    M1 = sysParams.M1;
    M2 = sysParams.M2;

    % solve the Lagrange equation F - fc = M*qdd + V*qd + G
    % compute qdd: M*qdd = F - fc - V*qd - G, using linsolve
    A = [M1+M2 M2*L*cos(q2); M2*L*cos(q2) M2*L^2];
    B = [F(1)-fc-C*q1d+M2*L*sin(q2)*q2d^2-K*q1; F(2)-M2*G*L*sin(q2)];
    qdd = linsolve(A,B);

    xdot = zeros(4,1);
    xdot(1) = q1d;
    xdot(2) = qdd(1);
    xdot(3) = q2d;
    xdot(4) = qdd(2);
end