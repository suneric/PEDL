function fc = coulomb_friction(v, F)
    ctrlOptions = control_options();
    if ctrlOptions.friction == "none"
        fc = 0;
    else
        params = parameters();
        G = params(4);
        M1 = params(5);
        M2 = params(6);
        mu_s = params(7);
        mu_k = params(8);
        tolerance = params(9);
        N = (M1+M2)*G;
        % % fc = min(1,max(-1,1000*v))*mu_k*N;
        if abs(v) < tolerance
            fc = min(F,mu_s*N);
        else
            fc = mu_k*N*sign(v);
            % fc = min(1,max(-1,100*v))*mu_k*N;
        end
        %fc = min(1,max(-1,v))*mu_k*N;
    end
end