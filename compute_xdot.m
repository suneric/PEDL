function xdot = compute_xdot(x,F,fc)
    q1 = x(1);
    q1dot = x(2);
    q2 = x(3);
    q2dot = x(4);
    
    % system parameters
    params = parameters();
    K = params(1);
    C = params(2);
    L = params(3);
    G = params(4);
    M1 = params(5);
    M2 = params(6);

    % solve the Lagrange equation F = M*q_ddot + V*q_dot + G
    % compute q_ddot: M*q_ddot = F - V*q_dot - G, using linsolve
    A = [M1+M2 M2*L*cos(q2); M2*L*cos(q2) M2*L*L];
    B = [F(1)-fc-C*q1dot+M2*L*sin(q2)*q2dot*q2dot-K*q1; F(2)-M2*G*L*sin(q2)];
    qddot = linsolve(A,B);

    xdot = zeros(4,1);
    xdot(1) = q1dot;
    xdot(2) = qddot(1);
    xdot(3) = q2dot;
    xdot(4) = qddot(2);
end