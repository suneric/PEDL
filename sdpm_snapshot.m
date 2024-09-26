function sdpm_snapshot(sysParams, t, x, th, xp, thp, tSpan)
    idx = find(t <= tSpan(2), 1, 'last');
    Xcg = x(idx);
    theta = th(idx);
    Xcg_pred = xp(idx);
    theta_pred = thp(idx);
    
    idx0 = find(t <= tSpan(1), 1, 'last');
    Xcg0 = x(idx0);
    theta0 = th(idx0);
    Xcg_pred0 = xp(idx0);
    theta_pred0 = thp(idx0);

    % Animation
    Ycg = 0;
    % plot limits
    Xmin = -5;
    Xmax = 5;
    Ymin = -1;
    Ymax = 2;
    l = sysParams.L; % pendulum rod length
    cartHalfLen = 0.7;
    
    f = figure('Color', 'White');
    f.Position = [500 200 800 500];
    hold on
    % Plot one frame...
    [h1,h2]=sdpm_plot_frame(Ycg,Xmin,Xmax,Ymax,l,cartHalfLen,Xcg,theta,Xcg_pred,theta_pred);

    % System initial state
    patch('XData', Xcg0+[-cartHalfLen cartHalfLen cartHalfLen -cartHalfLen],...
        'YData', Ycg+[cartHalfLen cartHalfLen -cartHalfLen -cartHalfLen],...
        'FaceColor','none', 'FaceAlpha', 0, ...
        'EdgeColor','k','LineWidth',1,'LineStyle','--');
    % plots pendulum
    h3 = plot([Xcg0 Xcg0+l*sin(theta0)],[Ycg Ycg-l*cos(theta0)],'k','LineWidth', 1, 'LineStyle','--', "DisplayName", "Initial Position"); 
    plot(Xcg0+l*sin(theta0),Ycg-l*cos(theta0),'Marker','o','MarkerSize',12,'MarkerEdgeColor','k'); 

    disErr = Xcg_pred - Xcg;
    angErr = theta_pred - theta;
    annotation('textbox', [0.69, 0.38, 0.3, 0.2], ...
        'String', {"q1 error: "+num2str(disErr,'%.3f') + " m" , "q2 error: " + num2str(angErr,'%.3f') + " rad"}, ...
        'FitBoxToText', 'on', ...
        'BackgroundColor', 'white', ...
        'EdgeColor', 'White', ...
        'FontName', 'Arial', ...
        'FontSize', 15);
    
    axis([Xmin Xmax Ymin Ymax])
    set(gca, "YTick", []);
    set(gca, "FontName", "Arial");
    set(gca, "FontSize", 12);
    xlabel("(m)", "FontSize", 15, "FontName","Arial")
    daspect([1 1 1])

    tObj = title("System at "+num2str(tSpan(2)-tSpan(1))+" second", "FontName", "Arial","FontSize",15);
    tObj.Position(1) = -3.0;
    legend([h1 h2 h3], "FontName","Arial", "FontSize", 15, 'Position', [0.7, 0.62, 0.2, 0.1]);
    
    saveas(f,'sdpm_snapshot.jpg')
end