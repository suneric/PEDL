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
    f.Position = [500 500 800 500];
    hold on

    % Plot one frame...
    plot([Xmin Xmin Xmax],[Ymax Ycg-cartHalfLen Ycg-cartHalfLen],'k','LineWidth',3) % ground and wall.

    % True system
    % cart
    patch(Xcg+[-cartHalfLen cartHalfLen cartHalfLen -cartHalfLen], ...
        Ycg+[cartHalfLen cartHalfLen -cartHalfLen -cartHalfLen],'k','FaceAlpha', 0, 'LineWidth',2); 
    
    % spring
    springX = zeros(10,1);
    springX(1) = Xmin;
    springX(2) = Xmin+1;
    springX(3:8) = linspace(Xmin+1,Xcg-cartHalfLen-1,6);
    springX(9) = Xcg-cartHalfLen-1;
    springX(10) = Xcg-cartHalfLen;
    springY = Ycg+cartHalfLen/2+[0 0 .15 -.15 .15 -.15 .15 -.15 0 0];
    plot(springX, springY,'b','LineWidth',1) 

    %  damper
    plot([Xmin Xcg-3],Ycg-cartHalfLen/2+[0 0],'b', ...
        [Xcg-3 Xcg-3],Ycg-cartHalfLen/2+[.15 -.15],'b',...
        [Xmin+1, Xcg-2, Xcg-2, Xmin+1],Ycg-cartHalfLen/2+[.15 .15 -.15,-.15],'b',...    
        [Xcg-2, Xcg-cartHalfLen],Ycg-cartHalfLen/2+[0 0],'b','LineWidth',1) 

    % plots rod and blob
    h1 = plot([Xcg Xcg+l*sin(theta)],[Ycg Ycg-l*cos(theta)],'k','LineWidth',3,'LineStyle','-', "DisplayName", "Reference"); 
    plot(Xcg+l*sin(theta),Ycg-l*cos(theta),'Marker','o','MarkerSize',12,'MarkerFaceColor','k','MarkerEdgeColor','k'); 
    
    % System predicted by model
    % cart
    patch('XData', Xcg_pred+[-cartHalfLen cartHalfLen cartHalfLen -cartHalfLen],...
        'YData', Ycg+[cartHalfLen cartHalfLen -cartHalfLen -cartHalfLen],...
        'FaceColor','none', 'FaceAlpha', 0, ...
        'EdgeColor','r','LineWidth',2,'LineStyle','--');
    % plots pendulum
    h2 = plot([Xcg_pred Xcg_pred+l*sin(theta_pred)],[Ycg Ycg-l*cos(theta_pred)],'r','LineWidth', 2, 'LineStyle','--',"DisplayName", "Prediction"); 
    plot(Xcg_pred+l*sin(theta_pred),Ycg-l*cos(theta_pred),'Marker','o','MarkerSize',12,'MarkerEdgeColor','r'); 

    % System initial state
    patch('XData', Xcg0+[-cartHalfLen cartHalfLen cartHalfLen -cartHalfLen],...
        'YData', Ycg+[cartHalfLen cartHalfLen -cartHalfLen -cartHalfLen],...
        'FaceColor','none', 'FaceAlpha', 0, ...
        'EdgeColor','k','LineWidth',1,'LineStyle','--');
    % plots pendulum
    h3 = plot([Xcg0 Xcg0+l*sin(theta0)],[Ycg Ycg-l*cos(theta0)],'k','LineWidth', 1, 'LineStyle','--', "DisplayName", "Initial Position"); 
    plot(Xcg0+l*sin(theta0),Ycg-l*cos(theta0),'Marker','o','MarkerSize',12,'MarkerEdgeColor','k'); 
    
    axis([Xmin Xmax Ymin Ymax])
    set(gca, "YTick", []);
    set(gca, "FontName", "Arial");
    set(gca, "FontSize", 12);
    xlabel("(m)", "FontSize", 15, "FontName","Arial")
    daspect([1 1 1])
    % titletext = {"Spring Damper Pendulum Mass system", "Mass Displacement: " + num2str(Xcg) + "      Predicted: " + num2str(Xcg_pred),...
    %         "Pendulum Angle: " + num2str(theta*180/pi) + "      Predicted: " + num2str(theta_pred*180/pi)};
    % title(titletext)
    tObj = title("System after "+num2str(tSpan(2)-tSpan(1))+" seconds", "FontName", "Arial","FontSize",15);
    tObj.Position(1) = -3.0;
    legend([h1 h2 h3], "FontName","Arial", "FontSize", 15, 'Position', [0.7, 0.62, 0.2, 0.1]);
    
    disErr = Xcg_pred - Xcg;
    angErr = theta_pred - theta;

    annotation('textbox', [0.69, 0.38, 0.3, 0.2], ...
        'String', {"q1 error: "+num2str(disErr,'%.3f') + " m" , "q2 error: " + num2str(angErr,'%.3f') + " rad"}, ...
        'FitBoxToText', 'on', ...
        'BackgroundColor', 'white', ...
        'EdgeColor', 'White', ...
        'FontName', 'Arial', ...
        'FontSize', 15);
    saveas(f,'sdpm_snapshot.jpg')
end