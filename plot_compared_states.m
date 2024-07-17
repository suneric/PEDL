function plot_compared_states(t,x,tp,xp)
    labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
    figure('Position',[500,100,800,800]);
    tiledlayout("vertical","TileSpacing","tight")
    numState = size(xp);
    numState = numState(2);
    for i = 1:numState
        nexttile
        plot(t,x(:,i),'b-',tp,xp(:,i),'r--','LineWidth',2);
        hold on
        xline(1,'k--', 'LineWidth',1);
        ylabel(labels(i),"Interpreter","latex");
        set(get(gca,'ylabel'),'rotation',0);
        set(gca, 'FontSize', 15);
        set(gca, 'FontName', "Arial")
        if i == numState
            xlabel("Time (s)");
        end
    end 
    legend("Reference","Prediction","Location","eastoutside","FontName","Arial");
end