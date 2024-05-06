%%
close all;
clear;
clc;

sample_number = [100,200,300,400,500,600,700,800,900];
lossType = ["PgNN","PcNN","PiNN"];

err_mat = zeros(length(lossType),length(sample_number));
for i = 1:length(lossType)
    type = lossType(i);
    for j = 1:length(sample_number)
        number = sample_number(j);
        model = load("test/"+type+"_"+num2str(number)+".mat");
        err_mat(i,j) = mean(model.err_list,'all');
    end
end
disp(err_mat);

%% plot test RMSE vs number of training samples
figure('Position',[100,100,800,400]);
plot(sample_number, err_mat(1,:),'Color','red','LineWidth',2,LineStyle='-');
hold on
plot(sample_number, err_mat(2,:),'Color','black','LineWidth',2,LineStyle='-');
hold on
plot(sample_number, err_mat(3,:),'Color','blue','LineWidth',2,LineStyle='-');
hold on
scatter(sample_number,err_mat(1,:),'filled','red');
hold on
scatter(sample_number,err_mat(2,:),'filled','black');
hold on
scatter(sample_number,err_mat(3,:),'filled','blue');
title("Model Performance vs. Sample Size");
ylabel("RMSE");
xlabel("Number of Training Samples");
xticks([100,200,300,400,500,600,700,800,900]);
legend("\alpha = 0", "\alpha = 0.5", "\alpha = 1");

%% plot best model
best_pgnn = 200;
pgnn = load("test/PgNN_"+num2str(best_pgnn)+".mat");
pgnn_errs = pgnn.err_list;
pgnn_q1_err = zeros(30,50);
pgnn_q2_err = zeros(30,50);
pgnn_q1d_err = zeros(30,50);
pgnn_q2d_err = zeros(30,50);
pgnn_q1dd_err = zeros(30,50);
pgnn_q2dd_err = zeros(30,50);
for i = 1:30
    startIdx = 6*(i-1)+1;
    pgnn_q1_err(i,:) = pgnn_errs(startIdx,:);
    pgnn_q2_err(i,:) = pgnn_errs(startIdx+1,:);
    pgnn_q1d_err(i,:) = pgnn_errs(startIdx+2,:);
    pgnn_q2d_err(i,:) = pgnn_errs(startIdx+3,:);
    pgnn_q1dd_err(i,:) = pgnn_errs(startIdx+4,:);
    pgnn_q2dd_err(i,:) = pgnn_errs(startIdx+5,:);
end

best_pinn = 700;
pinn = load("test/PiNN_"+num2str(best_pinn)+".mat");
pinn_errs = pinn.err_list;
pinn_q1_err = zeros(30,50);
pinn_q2_err = zeros(30,50);
pinn_q1d_err = zeros(30,50);
pinn_q2d_err = zeros(30,50);
pinn_q1dd_err = zeros(30,50);
pinn_q2dd_err = zeros(30,50);
for i = 1:30
    startIdx = 6*(i-1)+1;
    pinn_q1_err(i,:) = pinn_errs(startIdx,:);
    pinn_q2_err(i,:) = pinn_errs(startIdx+1,:);
    pinn_q1d_err(i,:) = pinn_errs(startIdx+2,:);
    pinn_q2d_err(i,:) = pinn_errs(startIdx+3,:);
    pinn_q1dd_err(i,:) = pinn_errs(startIdx+4,:);
    pinn_q2dd_err(i,:) = pinn_errs(startIdx+5,:);
end

best_pcnn = 300;
pcnn = load("test/PcNN_"+num2str(best_pcnn)+".mat");
pcnn_errs = pcnn.err_list;
pcnn_q1_err = zeros(30,50);
pcnn_q2_err = zeros(30,50);
pcnn_q1d_err = zeros(30,50);
pcnn_q2d_err = zeros(30,50);
pcnn_q1dd_err = zeros(30,50);
pcnn_q2dd_err = zeros(30,50);
for i = 1:30
    startIdx = 6*(i-1)+1;
    pcnn_q1_err(i,:) = pcnn_errs(startIdx,:);
    pcnn_q2_err(i,:) = pcnn_errs(startIdx+1,:);
    pcnn_q1d_err(i,:) = pcnn_errs(startIdx+2,:);
    pcnn_q2d_err(i,:) = pcnn_errs(startIdx+3,:);
    pcnn_q1dd_err(i,:) = pcnn_errs(startIdx+4,:);
    pcnn_q2dd_err(i,:) = pcnn_errs(startIdx+5,:);
end

%%
figure('Position',[100,100,800,200]);
t = linspace(1,10,50);
pinn_err = 0.5*(mean(pinn_q1_err,1)+mean(pinn_q2_err,1));
pgnn_err = 0.5*(mean(pgnn_q1_err,1)+mean(pgnn_q2_err,1));
pcnn_err = 0.5*(mean(pcnn_q1_err,1)+mean(pcnn_q2_err,1));
plot(t,pgnn_err,'Color','red','LineWidth',2);
hold on
plot(t,pcnn_err,'Color','black','LineWidth',2);
hold on
plot(t,pinn_err,'Color','blue','LineWidth',2);
title("Displacement Prediction Errors");
ylabel("RMSE");
xlabel("Time (s)");
xticks([1,2,3,4,5,6,7,8,9,10]);
legend("\alpha = 0", "\alpha = 0.5", "\alpha = 1");

figure('Position',[100,100,800,200]);
t = linspace(1,10,50);
pinn_err = 0.5*(mean(pinn_q1d_err,1)+mean(pinn_q2d_err,1));
pgnn_err = 0.5*(mean(pgnn_q1d_err,1)+mean(pgnn_q2d_err,1));
pcnn_err = 0.5*(mean(pcnn_q1d_err,1)+mean(pcnn_q2d_err,1));
plot(t,pgnn_err,'Color','red','LineWidth',2);
hold on
plot(t,pcnn_err,'Color','black','LineWidth',2);
hold on
plot(t,pinn_err,'Color','blue','LineWidth',2);
title("Velocity Prediction Errors");
ylabel("RMSE");
xlabel("Time (s)");
xticks([1,2,3,4,5,6,7,8,9,10]);
legend("\alpha = 0", "\alpha = 0.5", "\alpha = 1");

figure('Position',[100,100,800,200]);
t = linspace(1,10,50);
pinn_err = 0.5*(mean(pinn_q1dd_err,1)+mean(pinn_q2dd_err,1));
pgnn_err = 0.5*(mean(pgnn_q1dd_err,1)+mean(pgnn_q2dd_err,1));
pcnn_err = 0.5*(mean(pcnn_q1dd_err,1)+mean(pcnn_q2dd_err,1));
plot(t,pgnn_err,'Color','red','LineWidth',2);
hold on
plot(t,pcnn_err,'Color','black','LineWidth',2);
hold on
plot(t,pinn_err,'Color','blue','LineWidth',2);
title("Acceleration Prediction Errors");
ylabel("RMSE");
xlabel("Time (s)");
xticks([1,2,3,4,5,6,7,8,9,10]);
legend("\alpha = 0", "\alpha = 0.5", "\alpha = 1");