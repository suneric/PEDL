function plot_compared_states_t(t,x,tp,xp,numState)
    labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
    figure('Position',[500,100,800,500],'Color','White');
    tiledlayout("vertical","TileSpacing","tight")
    for i = 1:numState
        nexttile
        plot(t,x(:,i),'k-',tp,xp(:,i),'r--','LineWidth',2);
        % hold on
        % xline(1,'k--', 'LineWidth',1);
        set(gca, 'FontSize', 12); % Set font size of ticks
        ylabel(labels(i),"Interpreter","latex", 'FontSize', 18);
        set(get(gca,'ylabel'),'rotation',0);
        set(gca, 'FontName', "Arial")
        box off;
        if i == numState
            xlabel("Time (s)",'FontSize', 15);
        end
    end 
    h = legend("Reference","Prediction","FontName","Arial",'FontSize',15);
    legend boxoff
    set(h, 'Position', [0.7 0.85 0.2 0.1]);
end