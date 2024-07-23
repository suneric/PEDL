function params = params_training()
    params = struct();
    params.type = "pinn"; % "dnn","lstm","pinn"
    params.sequenceStep = 4; % 1 for non-lstm, 4,8,16 
    params.alpha = 0.2; % [0,1] weight of data loss and physics loss
    params.numSamples = 200; % 100,200,300,400,500
    params.numLayers = 5; % [3,10]
    params.numNeurons = 128; % 32,64,128,256
    params.dropoutFactor = 0.1; % 0.1,0.2
    params.learningRate = 0.001; % 0.01,0.001,0.0001
    params.miniBatchSize = 256; % [50,300]
    params.numEpochs = 50;
end