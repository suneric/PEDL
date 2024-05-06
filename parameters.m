%% System parameters
% C = 2*sqrt(mK)
function params = parameters()
    K = 6;
    C = 1;
    L = 0.3;
    G = 9.8;
    M1 = 1;
    M2 = 0.5;
    params = [K,C,L,G,M1,M2];
end