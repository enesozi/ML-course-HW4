## Load data and split data into test and training, Initialize variables.
clear all; close all;
data = load('points2d.dat');
[M,~] = size(data);
hidden_layer_size = 5;
input_layer_size = 2;
num_labels = 3;
trainRatio = .7;
valRatio = .3;
testRatio = 0;
[trainSet,validSet,testSet] = divideset(data,M,trainRatio,valRatio,testRatio); 

## Initialize Theta weights with some random values   
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

## Normalize training and test sets
X = trainSet(:,1:end-1);
X = X./max(abs(X));
y = trainSet(:,end);
X_test = validSet(:,1:end-1);
X_test = X_test./max(abs(X_test));
y_test = validSet(:,end);

## Construct output matrices with some probabilistic values as .9 .1
targetValues = 0.1*ones(size(y, 1), num_labels);
    for n = 1: size(y, 1)
        targetValues(n, y(n) + 1) = 0.9;
    end
testTargets = 0.1*ones(size(y_test, 1), num_labels);
    for n = 1: size(y_test, 1)
        testTargets(n, y_test(n) + 1) = 0.9;
    end

## Calculate Test & Training error and returns new Theta weights
epochs = 500;
lambda = 0.3;    
[errors errorTest Theta1 Theta2] = costFunc(X,targetValues,initial_Theta1,...
                                          initial_Theta2,epochs,lambda,X_test,testTargets);
## Plot Error Values
figure;
hold on;
title('Error Rates');
plot(errors,'r');
plot(errorTest, 'k--');
legend ("Train Error",'Test Error');
xlabel('Epochs');
ylabel('Error');
hold off;
## Calculating and plotting confusion matrix
[conf] = confusionMatPlot(X_test,testTargets,Theta1,Theta2);
# Plots decision boundry
decisionBoundry(Theta1,Theta2);
