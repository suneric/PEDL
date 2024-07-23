function fc = coulomb_friction(v, sysParams, friction)
% return fc is a positive value
    G = sysParams.G;
    M1 = sysParams.M1;
    M2 = sysParams.M2;
    mu_s = sysParams.mu_s;  % static friction coefficient
    mu_k = sysParams.mu_k; % kinetic friction coefficient
    N = (M1+M2)*G; % Normal force
    switch friction
        case "none"
            fc = 0;
        case 'smooth'
            fc = smooth_model(mu_s, N, v);
        case 'andersson'
            fc = andersoon_model(mu_s, mu_k, N, v);
        otherwise
            fc = 0;
    end
end

function fc = smooth_model(mu_s, N, v)
    % disp("Apply smooth coulomb friction.")
    vd = 0.01; % m/s
    fc = mu_s * N * tanh(v/vd);
end

function fc = andersoon_model(mu_s, mu_k, N, v)
    % disp("Apply andersson coulomb friction.")
    vs = 0.01; % m/s
    k = 10; % transition steepness parameter
    mu_v = 0.05; % viscous friction coefficient
    p = 2; % stribeck curve shape parameter;
    mu = mu_k + (mu_k-mu_s) * exp(-(abs(v)/vs)^p); % friction coefficient
    fc = mu * N * tanh(k*v) + mu_v*v;
end