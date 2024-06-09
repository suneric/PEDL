classdef myRegressionLayer < nnet.layer.RegressionLayer ...
        & nnet.layer.Acceleratable
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = myRegressionLayer(name)
            % layer = myRegressionLayer(name) creates a
            % mean-sqaure-error regression layer and specifies the layer
            % name.
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = 'Mean Square Error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            % data loss: compute the difference between target and predicted values
            dataLoss = mean((T-Y).^2,'all');
            % physics loss
            f = physics_law(Y(1:2,:),Y(3:4,:),Y(5:6,:));
            physicLoss = mean(f.^2,'all');
            % final loss, combining data loss and physics loss
            ctrlOptions = control_options();
            loss = (1.0-ctrlOptions.alpha)*dataLoss + ctrlOptions.alpha*physicLoss;
        end

        function dLdY = backwardLoss(layer,Y,T)
            % (Optional) Backward propagate the derivative of the loss 
            % function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the 
            %                 predictions Y        

            dLdY = 2 * (Y - T) / numel(T);
        end
    end
end