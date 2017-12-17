function [errorTest] = nnPredict(X,y,Theta1,Theta2)
## Makes probabalistic assumption for X b using Theta weights.
  batchSize = size(X,1);
  inputValues = [ones(batchSize,1) X];
  errorTest = 0;
#Sum squared error 
  SSE = 0;
    # Forward Propogation with each test sample.
    for k=1:batchSize       
        inputVector = inputValues(k,:);
        hiddenActualInput = inputVector*Theta1; #[1*3]*[3*5] = [1*5]
        hiddenOutputVector = activation(hiddenActualInput); 
        hiddenBiasOutputVector = [1 hiddenOutputVector]; #[1*6]
        outputActualInput = hiddenBiasOutputVector*Theta2; #[1*3]
        targetVector = y(k, :); 
     # Calculating and summing up square errors.   
        SSE= SSE + sumsq(targetVector-outputActualInput)/2;  
    end
     # Returns mean square error.
    errorTest=SSE/batchSize;
end