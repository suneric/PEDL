function [h1,h2]=sdpm_plot_frame(Ycg,Xmin,Xmax,Ymax,l,cartHalfLen,Xcg,theta,Xcg_pred,theta_pred)
    % ground and wall
    plot([Xmin Xmin Xmax],[Ymax Ycg-cartHalfLen Ycg-cartHalfLen],'k','LineWidth',3) 

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
end