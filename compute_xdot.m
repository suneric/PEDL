function xdot = compute_xdot(x, F, fc, sysParams)
    q1 = x(1);
    q1dot = x(2);
    q2 = x(3);
    q2dot = x(4);
    
    % system parameters
    K = sysParams.K;
    C = sysParams.C;
    L = sysParams.L;
    G = sysParams.G;
    M1 = sysParams.M1;
    M2 = sysParams.M2;

    % solve the Lagrange equation F - fc = M*q_ddot + V*q_dot + G
    % compute q_ddot: M*q_ddot = F - fc - V*q_dot - G, using linsolve
    A = [M1+M2 M2*L*cos(q2); M2*L*cos(q2) M2*L*L];
    B = [F(1)-fc-C*q1dot+M2*L*sin(q2)*q2dot*q2dot-K*q1; F(2)-M2*G*L*sin(q2)];
    qddot = linsolve(A,B);

    xdot = zeros(4,1);
    xdot(1) = q1dot;
    xdot(2) = qddot(1);
    xdot(3) = q2dot;
    xdot(4) = qddot(2);
end