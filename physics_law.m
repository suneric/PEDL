function YF = physics_law(X)
    % system parameters
    params = parameters();
    K = params(1);
    C = params(2);
    L = params(3);
    G = params(4);
    M1 = params(5);
    M2 = params(6);

    q1 = X(1,:);
    q2 = X(2,:);
    q1d = X(3,:);
    q2d = X(4,:);
    q1dd = X(5,:);
    q2dd = X(6,:);

    f1 = (M1+M2)*q1dd + M2*L*cos(q2).*q2dd + C*q1d + M2*L*sin(q2).*q2d.^2 - K*q1;
    f2 = M2*L*cos(q2).*q1dd + M2*L*L*q2dd + M2*G*L*sin(q2);
    YF = [f1;f2];
end