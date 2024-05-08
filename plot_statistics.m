%%
close all;
clear;
clc;
disp("clear and start the program")

%% set task type
task = "predict_next";
seqSteps = 20;
tForceStop = 1;% time stop force
tSpan = [0,10]; % simulation time span
tTest = [1,10]; % prediction test time span
ctrlOptions = control_options();
maxForces = linspace(0.5,15,30);
disp("initialize parameters")

%%
numSamples = [100,200,300,400,500,600,700,800,900];
rmse_pgnn = zeros(1,length(numSamples));
rmse_pcnn = zeros(1,length(numSamples));
rmse_pinn = zeros(1,length(numSamples));
for i = 1:length(numSamples)
    errs = load_errs(numSamples(i),"PgNN",maxForces,ctrlOptions,seqSteps,tForceStop,tTest,tSpan,task);
    rmse_pgnn(1,i) = mean(errs,"all");
    errs = load_errs(numSamples(i),"PcNN",maxForces,ctrlOptions,seqSteps,tForceStop,tTest,tSpan,task);
    rmse_pcnn(1,i) = mean(errs,"all");
    errs = load_errs(numSamples(i),"PiNN",maxForces,ctrlOptions,seqSteps,tForceStop,tTest,tSpan,task);
    rmse_pinn(1,i) = mean(errs,"all");
end
plot_comparison(numSamples,rmse_pgnn,rmse_pcnn,rmse_pinn);
disp("load data")

%% Average prediction accuracy
best_pgnn = load("model/PgNN_model_900.mat").net;
best_pcnn = load("model/PcNN_model_300.mat").net;
best_pinn = load("model/PiNN_model_700.mat").net;
tRef = linspace(1,10,100);
errs_pgnn = zeros(length(maxForces),100);
errs_pcnn = zeros(length(maxForces),100);
errs_pinn = zeros(length(maxForces),100);
for i = 1:length(maxForces)
    ctrlOptions.fMax = [maxForces(i);0];
    y = sdpm_simulation(tSpan, ctrlOptions);
    t = y(:,1);
    u = y(:,2:3);
    x = y(:,4:9);
    x_pgnn = predict_motion(best_pgnn,t,x,u,seqSteps,tForceStop,task);
    x_pcnn = predict_motion(best_pcnn,t,x,u,seqSteps,tForceStop,task);
    x_pinn = predict_motion(best_pinn,t,x,u,seqSteps,tForceStop,task);
    tTestIndices = zeros(1,100);
    for j = 1:100
        indices = find(t<=tRef(j));
        tTestIndices(1,j) = indices(end);
    end
    pg_rse = root_square_err(tTestIndices,x,x_pgnn);
    errs_pgnn(i,:) = mean(pg_rse,1);
    pc_rse = root_square_err(tTestIndices,x,x_pcnn);
    errs_pcnn(i,:) = mean(pc_rse,1);
    pi_rse = root_square_err(tTestIndices,x,x_pinn);
    errs_pinn(i,:) = mean(pi_rse,1);
end
plot_best(tTest,errs_pgnn,errs_pcnn,errs_pinn);
disp("average prediction accuracy")

%% Single case prediction accuracy over specified time span
best_pgnn = load("model/PgNN_model_800.mat").net;
best_pcnn = load("model/PcNN_model_300.mat").net;
best_pinn = load("model/PiNN_model_700.mat").net;
ctrlOptions.fMax = [12;0];
y = sdpm_simulation(tSpan, ctrlOptions);
t = y(:,1);
u = y(:,2:3);
x = y(:,4:9);
tTest = [5,10];
indices = find(t >= tTest(1) & t <= tTest(end));
x_pgnn = predict_motion(best_pgnn,t,x,u,seqSteps,tForceStop,task);
x_pcnn = predict_motion(best_pcnn,t,x,u,seqSteps,tForceStop,task);
x_pinn = predict_motion(best_pinn,t,x,u,seqSteps,tForceStop,task);
pg_rse = root_square_err(indices,x,x_pgnn);
pc_rse = root_square_err(indices,x,x_pcnn);
pi_rse = root_square_err(indices,x,x_pinn);
pg_rmse = mean(pg_rse,"all");
pc_rmse = mean(pc_rse,"all");
pi_rmse = mean(pi_rse,"all");
disp(pg_rmse);
disp(pc_rmse);
disp(pi_rmse);
disp("Single case predition accuracy")

%%
function plot_best(tTest,err_pgnn,err_pcnn,err_pinn)
    figure('Position',[500,100,800,300]);
    t = linspace(tTest(1),tTest(end),length(err_pgnn));
    plot(t,mean(err_pgnn,1),'Color','black','LineWidth',2,'LineStyle','-');
    hold on
    plot(t,mean(err_pcnn,1),'Color','blue','LineWidth',2,'LineStyle','-');
    hold on
    plot(t,mean(err_pinn,1),'Color','red','LineWidth',2,'LineStyle','-');
    xlabel("Time (s)","FontSize",20,"FontName","Arial");
    ylabel("RMSE","FontSize",20,"FontName","Arial");
    legend("\alpha = 0","\alpha = 0.5","\alpha = 1","Location",'best',"FontSize",12,"FontName","Arial");
    xticks([1,2,3,4,5,6,7,8,9,10]);
end

function plot_comparison(numSamples,err_pgnn,err_pcnn,err_pinn)
    figure('Position',[500,100,800,300]);
    plot(numSamples,err_pgnn,'Color','black','LineWidth',2,'LineStyle','-');
    hold on
    plot(numSamples,err_pcnn,'Color','blue','LineWidth',2,'LineStyle','-');
    hold on
    plot(numSamples,err_pinn,'Color','red','LineWidth',2,'LineStyle','-');
    hold on
    scatter(numSamples,err_pgnn,'filled','black')
    hold on
    scatter(numSamples,err_pcnn,'filled','blue')
    hold on
    scatter(numSamples,err_pinn,'filled','red')

    xlabel("Sample Size","FontSize",20,"FontName","Arial");
    ylabel("RMSE","FontSize",20,"FontName","Arial");
    legend("\alpha = 0","\alpha = 0.5","\alpha = 1","Location","best","FontSize",12,"FontName","Arial");
end

function errs = load_errs(sampleSize,type,maxForces,ctrlOptions,seqSteps,tForceStop,tTest,tSpan,task)
    fname = "test/"+type+"_"+num2str(sampleSize)+".mat";
    if exist(fname,'file') == 2
        errs = load(fname).err_list;
    else
        err_list = zeros(length(maxForces),6);
        model = load("model/"+type+"_model_"+num2str(sampleSize)+".mat").net;
        for i = 1:length(maxForces)
            ctrlOptions.fMax = [maxForces(i);0];
            y = sdpm_simulation(tSpan, ctrlOptions);
            t = y(:,1);
            u = y(:,2:3);
            x = y(:,4:9);
            indices = find(t >= tTest(1) & t <= tTest(end));
            x_pred = predict_motion(model,indices,x,u,seqSteps,tForceStop,task);
            rse = root_square_err(indices,x,x_pred);
            err_list(i,:) = mean(rse,2);
        end
        save(fname,'err_list');
        errs = err_list;
    end
end

function rse = root_square_err(indices,x,xPred)
    numPoints = length(indices);
    errs = zeros(6,numPoints);
    for i = 1:numPoints
        errs(1,i) = x(indices(i),1)-xPred(indices(i),1);
        errs(2,i) = x(indices(i),2)-xPred(indices(i),2);
        errs(3,i) = x(indices(i),3)-xPred(indices(i),3);
        errs(4,i) = x(indices(i),4)-xPred(indices(i),4);
        errs(5,i) = x(indices(i),5)-xPred(indices(i),5);
        errs(6,i) = x(indices(i),6)-xPred(indices(i),6);
    end
    rse = sqrt(errs.^2);
end

