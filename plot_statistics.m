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
disp("initialize parameters");

best_pgnn = load("model/PgNN_model_500.mat").net;
best_pcnn = load("model/PcNN_model_500.mat").net;
best_pinn = load("model/PiNN_model_600.mat").net;
disp("load best models");

%% plot accuracy pgnn and pinn each motion state comparison
numCase = 30;
numTime = 10;
refTime = linspace(1,10,numTime);
maxForces = linspace(0.5,15,numCase);
errs_pgnn = zeros(6*numCase,numTime);
errs_pinn = zeros(6*numCase,numTime);
for i = 1:numCase
    ctrlOptions.fMax = [maxForces(i);0];
    y = sdpm_simulation(tSpan, ctrlOptions);
    t = y(:,1);
    u = y(:,2:3);
    x = y(:,4:9);
    x_pgnn = predict_motion(best_pgnn,t,x,u,seqSteps,tForceStop,task);
    x_pinn = predict_motion(best_pinn,t,x,u,seqSteps,tForceStop,task);
    tTestIndices = zeros(1,numTime);
    for j = 1:numTime
        indices = find(t<=refTime(j));
        tTestIndices(1,j) = indices(end);
    end
    errs = root_square_err(tTestIndices,x,x_pgnn);
    idx = 6*(i-1);
    errs_pgnn(idx+1,:) = errs(1,:);
    errs_pgnn(idx+2,:) = errs(2,:);
    errs_pgnn(idx+3,:) = errs(3,:);
    errs_pgnn(idx+4,:) = errs(4,:);
    errs_pgnn(idx+5,:) = errs(5,:);
    errs_pgnn(idx+6,:) = errs(6,:);
    errs = root_square_err(tTestIndices,x,x_pinn);
    idx = 6*(i-1);
    errs_pinn(idx+1,:) = errs(1,:);
    errs_pinn(idx+2,:) = errs(2,:);
    errs_pinn(idx+3,:) = errs(3,:);
    errs_pinn(idx+4,:) = errs(4,:);
    errs_pinn(idx+5,:) = errs(5,:);
    errs_pinn(idx+6,:) = errs(6,:);
    
end

disp(["pgnn",mean(errs_pgnn,1)])
disp(["pinn",mean(errs_pinn,1)])

disp("plot time step rsme")
figure('Position',[500,100,800,300]); 
tiledlayout("vertical","TileSpacing","tight")
plot(refTime,mean(errs_pgnn,1),'k-',refTime,mean(errs_pinn,1),'r-','LineWidth',2);
xlabel("Time (s)","FontName","Arial");
ylabel("Average RMSE","FontName","Arial");
legend("\alpha = 0","\alpha = 1","Location",'best',"FontName","Arial");
xticks([1,2,3,4,5,6,7,8,9,10]);
set(gca, 'FontSize', 15);

%% Average prediction time
model = best_pinn;
% simulation with small time step
tSpan = [0,10];
ctrlOptions.fMax = [8;0];
tic;
y = sdpm_simulation(tSpan, ctrlOptions);
t_ode = toc;
t = y(:,1);
u = y(:,2:3);
x = y(:,4:9);

% predict with big time step
dTime = 0.03;
tp = 1:dTime:tSpan(end);
t_pred = zeros(seqSteps+length(tp),1);
x_pred = zeros(seqSteps+length(tp),6);
u_pred = zeros(seqSteps+length(tp),2);

initSteps = tForceStop-seqSteps*dTime:dTime:tForceStop;
for i = length(initSteps)-seqSteps:length(initSteps)-1
    indices = find(t < initSteps(i));
    t_pred(i) = t(indices(end));
    x_pred(i,:) = x(indices(end),:);
    u_pred(i,:) = u(indices(end),:);
end
t_pred(seqSteps+1:end) = tp;

dnn_total = 0;
for i = seqSteps+1:length(t_pred)-1
    startIdx = i-seqSteps+1;
    endIdx = i;
    nextIdx = i+1;
    state = {[t_pred(startIdx:endIdx),x_pred(startIdx:endIdx,:),u_pred(startIdx:endIdx,:)]'};
    dsState = arrayDatastore(state,'OutputType',"same",'ReadSize',128);
    dsTime = arrayDatastore(dTime,'ReadSize',128);
    dsTest = combine(dsState, dsTime);
    tic
    x_pred(nextIdx,:) = predict(model,dsTest);
    t_dnn = toc;
    dnn_total = dnn_total+t_dnn;
end

t_dnn = toc;
disp(["ode:",t_ode]);
disp(["dnn:",dnn_total]);

labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
figure('Position',[500,100,800,800]);
tiledlayout("vertical","TileSpacing","tight")
for i = 1:6
    nexttile
    plot(t,x(:,i),'b-',t_pred,x_pred(:,i),'r--','LineWidth',2);
    hold on
    xline(1,'k--', 'LineWidth',1);
    ylabel(labels(i),"Interpreter","latex","FontName","Arial");
    set(get(gca,'ylabel'),'rotation',0);
    if i == 6
        xlabel("Time (s)","Interpreter","latex","FontName","Arial");
    end
    if i == 1
        legend("Reference","Prediction","Location","best","FontName","Arial");
    end
end 
set(gca, 'FontSize', 15);


%% Single case prediction accuracy over specified time span
ctrlOptions.fMax = [3;0];
y = sdpm_simulation(tSpan, ctrlOptions);
t = y(:,1);
u = y(:,2:3);
x = y(:,4:9);
x_pgnn = predict_motion(best_pgnn,t,x,u,seqSteps,tForceStop,task);
x_pinn = predict_motion(best_pinn,t,x,u,seqSteps,tForceStop,task);

tTest = [1,5];
indices = find(t >= tTest(1) & t <= tTest(end));
pg_rse = root_square_err(indices,x,x_pgnn);
pi_rse = root_square_err(indices,x,x_pinn);
pg_rmse = mean(pg_rse,"all");
pi_rmse = mean(pi_rse,"all");
disp(pg_rmse);
disp(pi_rmse);
disp("Single case predition accuracy")

figure('Position',[500,100,1000,800]);
labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
tiledlayout("vertical","TileSpacing","tight")
for i = 1:6
    nexttile
    plot(t,x(:,i),'b-',t,x_pgnn(:,i),'k--',t,x_pinn(:,i),'r--','LineWidth',2);
    hold on
    xline(1,'k--', 'LineWidth',1);
    ylabel(labels(i),"Interpreter","latex");
    xticks([])
    if i == 6
        xlabel("Time (s)");
        xticks([1,2,3,4,5,6,7,8,9,10])
    end
    if i == 1
        legend("Reference","Prediction (\alpha = 0)","Prediction (\alpha = 1)","Location","northeastoutside");
    end
    set(get(gca,'ylabel'),'rotation',0);
    set(gca, 'FontSize', 15);
    set(gca, 'FontName', "Arial");
end 


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

%%
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
            x_pred = predict_motion(model,t,x,u,seqSteps,tForceStop,task);
            indices = find(t >= tTest(1) & t <= tTest(end));
            rse = root_square_err(indices,x,x_pred);
            err_list(i,:) = mean(rse,2);
        end
        save(fname,'err_list');
        errs = err_list;
    end
end

function rse = root_square_err(indices,x,xPred)
    numPoints = length(indices);
    x_size = size(xPred);
    errs = zeros(x_size(2),numPoints);
    for i = 1:numPoints
        for j = 1:x_size(2)
            errs(j,i) = x(indices(i),j)-xPred(indices(i),j);
        end
    end
    rse = sqrt(errs.^2);
end

