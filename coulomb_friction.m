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
        case 'specker'
            fc = specker_model(mu_s, mu_k, N, v);
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
    vd = 0.1; % m/s
    k = 10000; % transition steepness parameter
    p = 2; % stribeck curve shape parameter;
    % tanh(kv) term smoothly transitions from static friction to kinetic friction
    % when v is small, tanh(kv) = kv, when v is large tanh(kv) = 1.
    % this helps to model the gradual increase in friction forces as
    % velocity increases from zero.
    fc = N*(mu_k+(mu_s-mu_k)*exp(-(abs(v)/vd).^p)).*tanh(k*v); 
end

function fc = specker_model(mu_s, mu_k, N, v)
    vd = 0.05; % m/s
    vt = 2*vd;
    kv = 0;
    fc = (N*mu_s-N*mu_k*tanh(vt/vd)-kv*vt).*(v/vt).*exp(0.5*(1-(v/vt).^2)) + N*mu_k*tanh(v/vd);
end