function fc = coulomb_friction(v, F)
    params = parameters();
    G = params(4);
    M1 = params(5);
    M2 = params(6);
    N = (M1+M2)*G;
    mu_s = params(7);  
    mu_k = params(8);
    ctrlOptions = control_options();
    switch ctrlOptions.friction
        case "none"
            fc = 0;
        case 'smooth'
            vd = 0.01; % m/s
            fc = mu_s*N*tanh(v/vd);    
        case 'coulomb'       
            tolerance = params(9);
            if abs(v) < tolerance
                fc = min(F,mu_s*N);
            else
                fc = mu_k*N*sign(v);
            end
        otherwise
            fc = min(1,max(-1,1000*v))*mu_s*N;
    end
end