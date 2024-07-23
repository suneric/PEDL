function params = params_system()
    params = struct();
    params.K = 6; % spring coefficient
    params.C = 1; % damping coefficient
    params.L = 0.5; % length of pendulum
    params.G = 9.8; % gravity
    params.M1 = 1; % point mass of box
    params.M2 = 0.5; % point mass of pendulum
    params.mu_s = 0.15; % static friction 
    params.mu_k = 0.1; % kinetic friction
end