function [errors errorsTest Theta1 Theta2] = costFunc(X,y,Theta1,Theta2,epochs,...
                                                      learning_rate,Xtest,yTest)
## Calculates errors for training and test set simultaneously.
                             
batchSize = size(X,1);
inputValues = [ones(batchSize,1) X];
errors = zeros(epochs,1);
errorsTest = zeros(epochs,1);
# Iterate over epochs number
for t =1:epochs
    # Sum squared error 
    SSE=0;
    # Update for Theta weights.
    deltaT2=zeros(6,3); # Btw hidden-output
    deltaT1=zeros(3,5); # Btw input-hidden
    randIndc = randperm(batchSize);
    # Randomly iterate over the batch.
    for k=randIndc(1:batchSize)       
        # Forwar Propogation for a sample. 
        inputVector = inputValues(k,:);
        hiddenActualInput = inputVector*Theta1; #[1*3]*[3*5] = [1*5]
        hiddenOutputVector = activation(hiddenActualInput); 
        hiddenBiasOutputVector = [1 hiddenOutputVector]; #[1*6]
        outputActualInput = hiddenBiasOutputVector*Theta2; #[1*3]
        targetVector = y(k, :);    
        # Sum up squared error.
        SSE=SSE+sumsq(targetVector-outputActualInput)/2;      
        
        % Backpropagate the errors.
        oDelta  = hiddenBiasOutputVector'*(targetVector-outputActualInput); #[6*1]* [1*3]
        deltaT2 =  deltaT2+learning_rate*oDelta;            
        hDelta  =  (sum((targetVector-outputActualInput).* Theta2).*inputVector)'*dActivation(hiddenOutputVector); 
        deltaT1 = deltaT1+learning_rate*hDelta;     
      
    end
    # Error calculating for test set with updated Theta values.
    [errorTest] = nnPredict(Xtest,yTest,Theta1,Theta2);
    # Updating Theta
    Theta1 = Theta1+deltaT1/batchSize;
    Theta2 = Theta2+deltaT2/batchSize;
    # Mean square error for training set.
    MeanSquareError=SSE/batchSize;
    # Keeping the errors for each epoch.
    errors(t)=MeanSquareError;
    errorsTest(t) = errorTest;
end
        
 