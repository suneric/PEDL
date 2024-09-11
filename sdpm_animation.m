function sdpm_animation(sysParams, t, x, th, xp, thp)
     % Set up video
    v=VideoWriter('sdpm_animation.avi');
    v.FrameRate=60;
    open(v);
    
    % Animation
    Ycg = 0;
    % plot limits
    Xmin = -6;
    Xmax = 10;
    Ymin = -2;
    Ymax = 2;
    l = 4*sysParams.L; % pendulum rod length
    cartHalfLen = 1;
   
    f = figure('Color', 'White');
    f.Position = [500 500 800 500];
    
    for n = 1:length(t)
        cla
        Xcg = x(n);
        theta = th(n);
        Xcg_pred = xp(n);
        theta_pred = thp(n);

        % Plot everything for the video
        hold on
        % Plot one frame...
        % True system
        patch(Xcg+[-cartHalfLen cartHalfLen cartHalfLen -cartHalfLen], ...
        Ycg+[cartHalfLen cartHalfLen -cartHalfLen -cartHalfLen],'k','FaceAlpha', 0); 
    
        % spring
        springX = zeros(10,1);
        springX(1) = Xmin;
        springX(2) = Xmin+1;
        springX(3:8) = linspace(Xmin+1,Xcg-cartHalfLen-1,6);
        springX(9) = Xcg-cartHalfLen-1;
        springX(10) = Xcg-cartHalfLen;
        springY = Ycg+cartHalfLen/2+[0 0 .15 -.15 .15 -.15 .15 -.15 0 0];
        plot(springX, springY,'b','LineWidth',2) 
    
        %  damper
        plot([Xmin Xmin+2],Ycg-cartHalfLen/2+[0 0],'b', ...
            [Xmin+2 Xmin+2],Ycg-cartHalfLen/2+[.15 -.15],'b',...
            [Xmin+1, Xcg-2, Xcg-2, Xmin+1],Ycg-cartHalfLen/2+[.15 .15 -.15,-.15],'b',...    
            [Xcg-2, Xcg-cartHalfLen],Ycg-cartHalfLen/2+[0 0],'b','LineWidth',2) 
    
        % plots rod and blob
        plot([Xcg Xcg+l*sin(theta)],[Ycg Ycg-l*cos(theta)],'k','LineWidth',3,'LineStyle','-'); 
        plot(Xcg+l*sin(theta),Ycg-l*cos(theta),'Marker','o','MarkerSize',12,'MarkerFaceColor','k','MarkerEdgeColor','k'); 
        
        % System predicted by model
        % cart
        patch('XData', Xcg_pred+[-cartHalfLen cartHalfLen cartHalfLen -cartHalfLen],...
            'YData', Ycg+[cartHalfLen cartHalfLen -cartHalfLen -cartHalfLen],...
            'FaceColor','none', 'FaceAlpha', 0, ...
            'EdgeColor','r','LineWidth',2,'LineStyle','--');
    
        % plots pendulum
        plot([Xcg_pred Xcg_pred+l*sin(theta_pred)],[Ycg Ycg-l*cos(theta_pred)],'r','LineWidth', 2, 'LineStyle','--'); 
        plot(Xcg_pred+l*sin(theta_pred),Ycg-l*cos(theta_pred),'Marker','o','MarkerSize',12,'MarkerEdgeColor','r'); 
        
        axis([Xmin Xmax Ymin Ymax])
        set(gca, "YTick", [])
        xlabel("q1 (m)")
        daspect([1 1 1])
        % titletext = {"Spring Damper Pendulum Mass system", "Mass Displacement: " + num2str(Xcg) + "      Predicted: " + num2str(Xcg_pred),...
        %         "Pendulum Angle: " + num2str(theta*180/pi) + "      Predicted: " + num2str(theta_pred*180/pi)};
        % title(titletext)

        frame=getframe(gcf);
        writeVideo(v,frame);
end