function plot_friction(sysParams, v)
    G = sysParams.G;
    M1 = sysParams.M1;
    M2 = sysParams.M2;
    mu_s = sysParams.mu_s;  % static friction coefficient
    mu_k = sysParams.mu_k; % kinetic friction coefficient
    N = (M1+M2)*G; % Normal force
    vd = 0.1;
    k = 10000;
    p = [1;2;3];
    fc = andersson_model(v, N, mu_k, mu_s, vd, k, p);
    figure('Position',[500,100,800,400],"Color","White");
    plot(v, fc(1,:), 'LineWidth', 2, 'LineStyle','-',"DisplayName","p=1");
    hold on;
    plot(v, fc(2,:), 'LineWidth', 2, 'LineStyle','-.',"DisplayName","p=2");
    hold on;
    plot(v, fc(3,:), 'LineWidth', 2, 'LineStyle','--',"DisplayName","p=3");
    xlabel('Velocity (m/s)');
    ylabel('Friction Force (N)');
    legend('Location','northeast');
    set(gca, 'FontSize', 15);
    set(gca, 'FontName', 'Arial');
end

function fc = andersson_model(v, N, mu_k, mu_s, vd, k, p)
    fc = N*(mu_k+(mu_s-mu_k).*exp(-(abs(v)/vd).^p)).*tanh(k*v); 
end