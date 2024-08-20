function params = params_training()
    params = struct();
    params.type = "pinn2"; % "dnn2","lstm2","pinn2","dnn4","lstm4","pinn4","dnn6", "lstm6","pinn6"
    params.sequenceStep = 4; % 1 for non-lstm, 4,8,16 
    params.numUnits = 8;  % number of LSTM units
    params.alpha = 0.5; % [0,1] weight of data loss and physics loss
    params.numSamples = 500; % 100,200,300,400,500
    params.numLayers = 8; % [3,10]
    params.numNeurons = 128; % 32,64,128,256
    params.dropoutFactor = 0; % 0.1,0.2
    params.miniBatchSize = 2000; % 
    params.numEpochs = 200;
    params.initTimeStep = 1;
    params.initLearningRate = 1e-3; % 0.01,0.001,0.0001
    params.stopLearningRate = 5e-5;
    params.lrDropFactor = 0.99;
    params.lrDropEpoch = 3;
end