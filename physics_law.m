function YF = physics_law(X)
    % system parameters
    params = parameters();
    K = params(1);
    C = params(2);
    L = params(3);
    G = params(4);
    M1 = params(5);
    M2 = params(6);

    N = size(X,2);
    if isa(X,'single')
        YF = zeros(2,N,'single');
    else
        YF = zeros(2,N,'single','gpuArray');
    end
    for i = 1:N
        q1 = X(1,i);
        q2 = X(2,i);
        q1dot = X(3,i);
        q2dot = X(4,i);
        q1ddot = X(5,i);
        q2ddot = X(6,i);
        A = [M1+M2 M2*L*cos(q2); M2*L*cos(q2) M2*L*L];
        B = [C*q1dot+M2*L*sin(q2)*q2dot*q2dot-K*q1; M2*G*L*sin(q2)];
        % solve the Lagrange equation F = M*q_ddot + V*q_dot + G 
        YF(:,i) = A*[q1ddot;q2ddot] + B;       
    end
end