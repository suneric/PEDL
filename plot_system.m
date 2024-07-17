function plot_system(sysParams, ctrlParams, fRange, tSpan)
    f1Min = max(15, sysParams.fc_max);
    ctrlParams.fMax = [f1Min+fRange; 0]; 
    y = sdpm_simulation(tSpan, sysParams, ctrlParams);
    t = y(:, 1);
    x = y(:, 2:7);
    f1 = y(:, 8);
    fc = y(:, 10);
    plot_states(t, x);
    plot_forces(t, f1, fc);
end

function plot_forces(t, f1, fc)
    figure('Position', [500,100,800,800]);
    plot(t, f1, 'k-', t, fc, 'b-', 'LineWidth', 2);
    legend("F1", "Fc");
end

function plot_states(t, x)
    refClr = "blue";
    labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
    figure('Position',[500,100,800,800]);
    tiledlayout("vertical","TileSpacing","tight")
    numState = size(x);
    numState = numState(2);
    for i = 1 : numState
        nexttile
        plot(t, x(:, i), 'Color', refClr, 'LineWidth', 2);
        hold on
        xline(1, 'k--', 'LineWidth', 2);
        ylabel(labels(i), "Interpreter", "latex");
        if i == numState
            xlabel("Time (s)");
        end
        set(get(gca, 'ylabel'), 'rotation', 0);
        set(gca, 'FontSize', 15);
        set(gca, 'FontName', 'Arial');
    end 
end