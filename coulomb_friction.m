function fc = coulomb_friction(v, sysParams, friction)
% return fc is a positive value
    G = sysParams.G;
    M1 = sysParams.M1;
    M2 = sysParams.M2;
    mu_s = sysParams.mu_s;  
    % mu_k = sysParams.mu_k;
    switch friction
        case "none"
            fc = 0;
        case 'smooth'
            vd = 0.01; % m/s
            fc = mu_s*(M1+M2)*G*tanh(v/vd);    
        otherwise
            fc = 0;
    end
end