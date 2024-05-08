function plot_states(t,x,x_pred,lossType)
    refClr = "cyan";
    predClr = "black";
    predStr = "Prediction";
    switch lossType
        case "PgNN"
            predClr = "black";
            predStr = "Prediction (\alpha = 0)";
        case "PcNN"
            predClr = "blue";
            predStr = "Prediction (\alpha = 0.5)";
        case "PiNN"
            predClr = "red";
            predStr = "Prediction (\alpha = 1)";
        otherwise
            predStr = "Prediction";
    end
    labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
    figure('Position',[500,100,800,800]);
    tiledlayout("vertical","TileSpacing","tight")
    for i = 1:6
        nexttile
        plot(t,x(:,i),'Color',refClr,'LineWidth',2);
        if length(x_pred) == length(x)
            hold on
            plot(t,x_pred(:,i),'Color',predClr,'LineWidth',2,'LineStyle','--');
        end
        hold on
        xline(1,'k--', 'LineWidth',1);
        ylabel(labels(i),"Interpreter","latex","FontSize",20,"FontName","Arial");
        set(get(gca,'ylabel'),'rotation',0);
        if i == 6
            xlabel("Time (s)","Interpreter","latex","FontSize",20,"FontName","Arial");
        end
        if i == 1
            legend("Reference",predStr,"Location","best","FontSize",12,"FontName","Arial");
        end
    end 
end