function sdpm_animation(sysParams, t, x, th, xp, thp, tSpan)
     % Set up video
    v=VideoWriter('sdpm_animation.avi');
    v.FrameRate=30;
    open(v);

    idx1 = find(t <= tSpan(1), 1, 'last');
    idx2 = find(t <= tSpan(2), 1, 'last');

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
    f.Position = [500 400 800 600];

    for n = idx1:idx2
        cla
        Xcg = x(n);
        theta = th(n);
        Xcg_pred = xp(n);
        theta_pred = thp(n);
        
        subplot(3,1,1)
        plot(t(idx1:n)-1,x(idx1:n),'k-',t(idx1:n)-1,xp(idx1:n),'r--','LineWidth',2);
        set(gca, 'FontSize', 12); % Set font size of ticks
        ylabel('$q_1$',"Interpreter","latex", 'FontSize', 18);
        set(get(gca,'ylabel'),'rotation',0);
        set(gca, 'FontName', "Arial")
        axis([0,max(t)-1 min(xp)-1 max(xp)+1])
        set(gca,'Position',[0.1,0.8,0.8,0.15]);
        grid on
        
        subplot(3,1,2)
        plot(t(idx1:n)-1,th(idx1:n),'k-',t(idx1:n)-1,thp(idx1:n),'r--','LineWidth',2);
        set(gca, 'FontSize', 12); % Set font size of ticks
        ylabel('$q_2$',"Interpreter","latex", 'FontSize', 18);
        set(get(gca,'ylabel'),'rotation',0);
        set(gca, 'FontName', "Arial")
        axis([0,max(t)-1 min(thp)-1 max(thp)+1])
        set(gca,'Position',[0.1,0.6,0.8,0.15]);
        xlabel("Time (s)",'FontSize', 15);
        set(gca, 'FontSize', 12); % Set font size of ticks
        grid on;
        
        subplot(3,1,3)
        hold on
        % Plot one frame...
        [h1,h2] = sdpm_plot_frame(Ycg,Xmin,Xmax,Ymax,l,cartHalfLen,Xcg,theta,Xcg_pred,theta_pred);

        disErr = Xcg_pred - Xcg;
        angErr = theta_pred - theta;
        annotation('textbox', [0.69, 0.2, 0.3, 0.2], ...
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
        set(gca,'Position',[0.1,0.1,0.8,0.4]);
    
        tObj = title("System at "+num2str(t(n)-1)+" second", "FontName", "Arial","FontSize",15);
        tObj.Position(1) = -3.0;
        legend([h1 h2], "FontName","Arial", "FontSize", 15, 'Position', [0.7, 0.4, 0.2, 0.1]);

        frame=getframe(gcf);
        writeVideo(v,frame);
    end
end