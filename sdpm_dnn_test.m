%%
close all;
clear;
clc;

%% set task type
lossType = "PiNN";
task = "predict_next"; % "predict_arbitrary";
seq_steps = 20;
tForceStop = 1;
num_samples = 500;
tSpan = [0,10];
% strType = {'constant','increase','decrease'};
ctrlOptions = control_options();
ctrlOptions.fMax = [8;0];
%ctrlOptions.fType = strType{randi(numel(strType))};
%ctrlOptions.fSpan = [0,randi([2,5])];

%% Test   
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
u = y(:,2:3);
x = y(:,4:9);
switch lossType
    case "PgNN"
        pgnn_model = load("model/PgNN_model_"+num2str(num_samples)+".mat").net;
        x_pgnn = predict_motion(pgnn_model,t,x,u,seq_steps,tForceStop,task);
        plot_states(t,x,x_pgnn,lossType);
    case "PcNN"
        pcnn_model = load("model/PcNN_model_"+num2str(num_samples)+".mat").net;
        x_pcnn = predict_motion(pcnn_model,t,x,u,seq_steps,tForceStop,task);
        plot_states(t,x,x_pcnn,lossType);
    case "PiNN"
        pinn_model = load("model/PiNN_model_"+num2str(num_samples)+".mat").net;
        x_pinn = predict_motion(pinn_model,t,x,u,seq_steps,tForceStop,task);
        plot_states(t,x,x_pinn,lossType);
    otherwise
        pgnn_model = load("model/PgNN_model_"+num2str(num_samples)+".mat").net;
        x_pgnn = predict_motion(pgnn_model,t,x,u,seq_steps,tForceStop,task);
        pcnn_model = load("model/PcNN_model_"+num2str(num_samples)+".mat").net;
        x_pcnn = predict_motion(pcnn_model,t,x,u,seq_steps,tForceStop,task);
        pinn_model = load("model/PiNN_model_"+num2str(num_samples)+".mat").net;
        x_pinn = predict_motion(pinn_model,t,x,u,seq_steps,tForceStop,task);
        plot_comparison(t,x,x_pgnn,x_pcnn,x_pinn);
end

%%
function plot_comparison(t,x,x_pgnn,x_pcnn,x_pinn)
    labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
    figure('Position',[500,100,800,800]);
    tiledlayout("vertical","TileSpacing","tight")
    for i = 1:6
        nexttile
        plot(t,x(:,i),'Color','cyan','LineWidth',2);
        hold on
        plot(t,x_pgnn(:,i),'Color','black','LineWidth',2,'LineStyle','--');
        hold on
        plot(t,x_pcnn(:,i),'Color','blue','LineWidth',2,'LineStyle','--');
        hold on
        plot(t,x_pinn(:,i),'Color','red','LineWidth',2,'LineStyle','--');
        hold on
        xline(1,'k--', 'LineWidth',1);
        ylabel(labels(i),"Interpreter","latex","FontSize",20,"FontName","Arial");
        set(get(gca,'ylabel'),'rotation',0);
        if i == 6
            xlabel("Time (s)","Interpreter","latex","FontSize",20,"FontName","Arial");
        end
    end
    legend("Reference","PgNN","PcNN","PiNN");
end
