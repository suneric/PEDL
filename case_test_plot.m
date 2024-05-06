%%
close all;
clear;
clc;

%% set task type
lossType = "PcNN";
task = "predict_next"; % task = "predict_arbitrary";
seq_steps = 20;
t_force_stop = 1;
tSpan = [0,10];
max_f1 = 6;
ctrlOptions = control_options();
ctrlOptions.fMax = [max_f1;0];

% Mass-Spring-Damper-Pendulum Dynamics System Parameters
y = sdpm_simulation(tSpan, ctrlOptions);
t = y(:,1);
u = y(:,2:3);
x = y(:,4:9);

pgnn_model = load("model/PgNN_model_"+num2str(200)+".mat").net;
pcnn_model = load("model/PcNN_model_"+num2str(300)+".mat").net;
pinn_model = load("model/PiNN_model_"+num2str(700)+".mat").net;

x_pgnn = predict_motion(pgnn_model,t,x,u,seq_steps,t_force_stop,task);
x_pcnn = predict_motion(pcnn_model,t,x,u,seq_steps,t_force_stop,task);
x_pinn = predict_motion(pinn_model,t,x,u,seq_steps,t_force_stop,task);
   
switch lossType
    case "PgNN"
        plot_states(t,x,ctrlOptions,x_pgnn);
    case "PcNN"
        plot_states(t,x,ctrlOptions,x_pcnn);
    case "PiNN"
        plot_states(t,x,ctrlOptions,x_pinn);
    otherwise
        plot_comparison(t,x,x_pgnn,x_pcnn,x_pinn);
end

%%
function plot_comparison(t,x,x_pgnn,x_pcnn,x_pinn)
    figure('Position',[100,100,800,600]);
    sgtitle("Displacement");
    subplot(2,1,1);
    plot(t,x(:,1),'Color','blue','LineWidth',2);
    xline(1,'k--', 'LineWidth',1);
    ylabel("$q_1$",'Interpreter','latex');
    set(get(gca,'ylabel'),'rotation',0);
    set(gca,'fontsize',12);
    hold on
    plot(t,x_pgnn(:,1),'Color','#C35048','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pcnn(:,1),'Color','#66C61C','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pinn(:,1),'Color','#1A4480','LineWidth',2,'LineStyle','--');

    subplot(2,1,2);
    plot(t,x(:,2),'Color','blue','LineWidth',2);
    xline(1,'k--', 'LineWidth',1);
    ylabel("$q_1$",'Interpreter','latex');
    set(get(gca,'ylabel'),'rotation',0);
    set(gca,'fontsize',12);
    hold on
    plot(t,x_pgnn(:,2),'Color','#C35048','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pcnn(:,2),'Color','#66C61C','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pinn(:,2),'Color','#1A4480','LineWidth',2,'LineStyle','--');
    legend("Reference","Prediction (\alpha = 0)", "Prediction (\alpha = 0.5)", "Prediction (\alpha = 1)");

    figure('Position',[100,100,800,600]);
    sgtitle("Velocity");
    subplot(2,1,1);
    plot(t,x(:,3),'Color','blue','LineWidth',2);
    xline(1,'k--', 'LineWidth',1);
    ylabel("$q_1$",'Interpreter','latex');
    set(get(gca,'ylabel'),'rotation',0);
    set(gca,'fontsize',12);
    hold on
    plot(t,x_pgnn(:,3),'Color','#C35048','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pcnn(:,3),'Color','#66C61C','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pinn(:,3),'Color','#1A4480','LineWidth',2,'LineStyle','--');

    subplot(2,1,2);
    plot(t,x(:,4),'Color','blue','LineWidth',2);
    xline(1,'k--', 'LineWidth',1);
    ylabel("$q_1$",'Interpreter','latex');
    set(get(gca,'ylabel'),'rotation',0);
    set(gca,'fontsize',12);
    hold on
    plot(t,x_pgnn(:,4),'Color','#C35048','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pcnn(:,4),'Color','#66C61C','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pinn(:,4),'Color','#1A4480','LineWidth',2,'LineStyle','--');
    legend("Reference","Prediction (\alpha = 0)", "Prediction (\alpha = 0.5)", "Prediction (\alpha = 1)");

    figure('Position',[100,100,800,600]);
    sgtitle("Acceleration");
    subplot(2,1,1);
    plot(t,x(:,5),'Color','blue','LineWidth',2);
    xline(1,'k--', 'LineWidth',1);
    ylabel("$\dot{q}_1$",'Interpreter','latex');
    set(get(gca,'ylabel'),'rotation',0);
    set(gca,'fontsize',12);
    hold on
    plot(t,x_pgnn(:,5),'Color','#C35048','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pcnn(:,5),'Color','#66C61C','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pinn(:,5),'Color','#1A4480','LineWidth',2,'LineStyle','--');

    subplot(2,1,2);
    plot(t,x(:,6),'Color','blue','LineWidth',2);
    xline(1,'k--', 'LineWidth',1);
    ylabel("$\dot{q}_1$",'Interpreter','latex');
    set(get(gca,'ylabel'),'rotation',0);
    set(gca,'fontsize',12);
    hold on
    plot(t,x_pgnn(:,6),'Color','#C35048','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pcnn(:,6),'Color','#66C61C','LineWidth',2,'LineStyle','--');
    hold on
    plot(t,x_pinn(:,6),'Color','#1A4480','LineWidth',2,'LineStyle','--');
    legend("Reference","Prediction (\alpha = 0)", "Prediction (\alpha = 0.5)", "Prediction (\alpha = 1)");
end