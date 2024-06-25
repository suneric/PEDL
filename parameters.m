%% System parameters
% C = 2*sqrt(mK)
function params = parameters()
    K = 6;
    C = 1;
    L = 0.5;
    G = 9.8;
    M1 = 1;
    M2 = 0.5;
    mu_s = 0.15;
    mu_k = 0.1;
    tolerance = 0.01;
    minF = max(mu_s,mu_k)*(M1+M2)*G;
    params = [K,C,L,G,M1,M2,mu_s,mu_k,tolerance,minF];
end