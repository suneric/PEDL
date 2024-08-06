function res = compare_model(folder, typeList, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, numTime, numState)
    % simulation baseline
    ctrlParams.fMax = [f1Max; 0]; 
    y = sdpm_simulation(tSpan, sysParams, ctrlParams);
    t = y(:,1);
    x = y(:,2:7);

    initIdx = find(t > ctrlParams.fSpan(2), 1, 'first');
    tp = t(initIdx:end);

    % load model and predict
    numModel = length(typeList);
    res = zeros(numModel, 1);
    xPreds = cell(numModel, 1);
    for i = 1:numModel
        type = typeList(i);
        modelFile = folder+"\"+type+"_"+num2str(trainParams.alpha)+"_"+num2str(trainParams.numSamples)+".mat";
        if exist(modelFile, 'file') == 2
            net = load(modelFile).net;
            [xp, rmseErr, ~] = evaluate_single(net, t, x, ctrlParams, trainParams, tSpan, predInterval, numTime, type);
            res(i) = mean(rmseErr,"all"); 
            xPreds{i} = xp;
        end
    end

    % plot
    labels = ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
    colors = ["#FF0000","#0000FF","#00A86B","#00FFFF","#FF00FF","#00FF00","#808080"];
    figure('Position',[500,200,800,600]);
    tiledlayout("vertical","TileSpacing","tight")
    h = [];
    for i = 1:numState
        nexttile
        h(1) = plot(t, x(:,i),'k-','LineWidth',2, 'DisplayName', 'Reference');
        hold on
        for j = 1:numModel
            xp = xPreds{j};
            xp = xp(initIdx:end,:);
            display = typeList(j) + " err: " + num2str(res(j));
            hj = plot(tp, xp(:,i), 'LineWidth', 2, 'DisplayName', display, 'Color', colors(j), 'LineStyle','--');
            if j+1 > length(h)
                h = [h,hj];
            end
            hold on
        end
        xline(1,'k--', 'LineWidth',1);
        ylabel(labels(i),"Interpreter","latex");
        set(get(gca,'ylabel'),'rotation',0);
        set(gca, 'FontSize', 15);
        set(gca, 'FontName', "Arial")
        if i == numState
            xlabel("Time (s)");
        end
    end 
    lh = legend(h,"FontName","Arial");
    set(lh, 'Position', [0.5, 0.45, 0.25, 0.15]);

end