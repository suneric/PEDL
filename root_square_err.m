% root square error of prediction and reference
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