function [ trainSet,validSet,testSet ] = divideset(data,M,trainRatio,valRatio,testRatio)
## Divides set into training and test sets with specified ratios randomly. 
   rIndices = randperm(M);
   trainSet = data(rIndices(1:M*trainRatio),:);
   validSet = data(rIndices(M*trainRatio+1:M*trainRatio+M*valRatio),:);
   testSet = data(rIndices(M*trainRatio+1+M*valRatio:end),:);
end

