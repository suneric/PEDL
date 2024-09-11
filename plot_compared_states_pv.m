function plot_compared_states_pv(x,xp)
    labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
    figure('Position',[500,200,800,300],'Color','white');
    tiledlayout("horizontal","TileSpacing","tight")
    
    nexttile
    plot(x(:,1),x(:,3),'k-',xp(:,1),xp(:,3),'r--','LineWidth',2);
    xlabel(labels(1),"Interpreter","latex");
    ylabel(labels(3),"Interpreter","latex");
    set(get(gca,'ylabel'),'rotation',0);
    set(gca, 'FontSize', 20);
    set(gca, 'FontName', "Arial")

    nexttile
    plot(x(:,2),x(:,4),'k-',xp(:,2),xp(:,4),'r--','LineWidth',2);
    xlabel(labels(2),"Interpreter","latex");
    ylabel(labels(4),"Interpreter","latex");
    set(get(gca,'ylabel'),'rotation',0);
    set(gca, 'FontSize', 20);
    set(gca, 'FontName', "Arial")

    h = legend("Reference","Prediction","FontName","Arial");
    set(h, 'Position', [0.6 0.7 0.2 0.1]);
end