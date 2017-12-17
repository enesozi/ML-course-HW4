function [confMatrix] = confusionMatPlot(X,y,Theta1,Theta2)
  ## Creates and plot confusion matrix 
  batchSize = size(X,1);
  inputValues = [ones(batchSize,1) X];
  confMatrix = zeros(size(y,2));
  tPfP = zeros(batchSize,1);
    # Iterates over the batch and makes forward propogate 
    for k=1:batchSize       
        inputVector = inputValues(k,:);
        hiddenActualInput = inputVector*Theta1; #[1*3]*[3*5] = [1*5]
        hiddenOutputVector = activation(hiddenActualInput); 
        hiddenBiasOutputVector = [1 hiddenOutputVector]; #[1*6]
        outputActualInput = hiddenBiasOutputVector*Theta2; #[1*3]
        targetVector = y(k, :);    
        # Class guessed and actual
        [~,pred] = max(outputActualInput);
        [~,class] = max(targetVector);
        confMatrix(pred,class) = confMatrix(pred,class) + 1;
        # if-else branches below are for storing values of TP & FP
        if class == 1
            if pred == 1
             tPfP(k) = 0; 
            else
             tPfP(k) = 1;
            end
        elseif class == 2
            if pred == 2
              tPfP(k) = 2;
            else
              tPfP(k) = 3;  
            end
        else
            if pred == 3
              tPfP(k) = 4;
            else 
              tPfP(k) = 5;
            end
        end  
    end
    
    # Plotting TP & FP for each class.
    figure;
    title('TP%FP Plots For Each Class');
    hold on;
    Xt = X(find(tPfP == 0),:);
    scatter(Xt(:,1),Xt(:,2),8,[1 0 0],'filled');
    Xt = X(find(tPfP == 1),:);
    scatter(Xt(:,1),Xt(:,2),8,[1 0 1],'filled');
    Xt = X(find(tPfP == 2),:);
    scatter(Xt(:,1),Xt(:,2),8,[0 0 1],'filled');
    Xt = X(find(tPfP == 3),:);
    scatter(Xt(:,1),Xt(:,2),8,[0 1 1],'filled');
    Xt = X(find(tPfP == 4),:);
    scatter(Xt(:,1),Xt(:,2),8,[1 1 0],'filled');
    Xt = X(find(tPfP == 5),:);
    scatter(Xt(:,1),Xt(:,2),8,[0 1 0],'filled');
    legend('TP0','FP0','TP1','FP1','TP2','FP2');
    hold off;
    
sA=size(confMatrix);

# Plotting confusion matrix as box
figure;
hold on;
rectangle('Position',[0,0,sA(2)+1,sA(1)+1],'Facecolor',[1 1 1],'edgecolor','none');

for ii=0:sA(1)+1
    plot([0 sA(2)+1], [ii ii],'k','Linewidth',3);
end
for ii=0:sA(2)+1
    plot([ii ii],[0 sA(1)+1],'k','Linewidth',3);
end
for ii=1:sA(1)
    for jj=1:sA(2)
    text((ii-1)+0.2,(3-jj)+0.5,num2str(confMatrix'(ii,jj)),'fontsize',15,'Color','k');
    end
end   
        # True positive ,false etc.  calculation
        # All code below is for printing rates in small boxes
        sumRow = sum(confMatrix,2)+1;
        sumCol = sum(confMatrix);
        diagS = diag(confMatrix);
        TP = 100*diagS'./sum(confMatrix);
        FN = 100*(sumRow-diag(confMatrix))./sumRow;
        FP = 100*(sumCol-diag(confMatrix)')./sumCol;
        
        for ii=1:3      
          text(ii-0.8,3.5,strcat(num2str(TP(ii)),' %'),'fontsize',15,'Color','g');
          text(ii-0.8,3.3,strcat(num2str(FP(ii)),' %'),'fontsize',15,'Color','r');
        end  
        for ii=1:3      
          text(3.2,ii-0.6,strcat(num2str(100-FN(ii)),' %'),'fontsize',15,'Color','g');
          text(3.2,ii-0.8,strcat(num2str(FN(ii)),' %'),'fontsize',15,'Color','r');
        end 
        text(3.2,3.5,strcat(num2str(100*sum(diagS)/batchSize),' %'),'fontsize',15,'Color','g');
        text(3.2,3.3,strcat(num2str(100-100*sum(diagS)/batchSize),' %'),'fontsize',15,'Color','r');
        xlabel('Actual class from 0-2');
        ylabel('Predicted class from 0-2');
        title('Confusion Matrix & TP-FP at top, TN-FN at right');
        hold off;
end

