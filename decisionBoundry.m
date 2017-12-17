function  decisionBoundry(Theta1,Theta2)
  # Init random matrix of size 20000*2 with range of 15:-15 
  r = (30).*rand(20000,2)-15;
  # Normalize r
  rN = r ./ max(r);
  batchSize = size(r,1);
  inputValues = [ones(batchSize,1) rN];
  # For recording classes guessed
  classes = zeros(batchSize,1);
  # Forward propogation and guessing class with probability.
    for k=1:batchSize       
        inputVector = inputValues(k,:);
        hiddenActualInput = inputVector*Theta1; #[1*3]*[3*5] = [1*5]
        hiddenOutputVector = activation(hiddenActualInput); 
        hiddenBiasOutputVector = [1 hiddenOutputVector]; #[1*6]
        outputActualInput = hiddenBiasOutputVector*Theta2; #[1*3]
        [~,pred] = max(outputActualInput);
        # Assign class.
        classes(k) = pred;
    end
    # Plot each class
    figure;
    hold on;
    X = r(find(classes == 1),:);
    scatter(X(:,1),X(:,2),'red');
    X = r(find(classes == 2),:);
    scatter(X(:,1),X(:,2),'blue');
    X = r(find(classes == 3),:);
    scatter(X(:,1),X(:,2),'green');
    legend('Class0','Class1','Class2');
    hold off;
end