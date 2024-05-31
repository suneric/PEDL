function fc = coulomb_friction(v, F)
    params = parameters();
    G = params(4);
    M1 = params(5);
    M2 = params(6);
    mu_s = params(7);
    mu_k = params(8);
    tolerance = params(9);
    N = (M1+M2)*G;
    if abs(v) < tolerance
        fc = min(F, mu_s*N);
    else
        fc = mu_k*N*sign(v);
    end
    %fc = min(1,max(-1,v))*mu_k*N;
end