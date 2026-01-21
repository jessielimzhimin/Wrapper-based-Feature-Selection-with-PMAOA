% Notation: This fitness function is for demonstration 

function [cost, error] = jFitnessFunction(feat,label,X,HO, thres)

%% To convert the real-value into binary value based on threshold
% X(d) is assigned as 1 ('feature is selected') if the original X(d) is larger than thres
% X(d) is assigned as 0 ('feature is not selected') if the original X(d) is smaller or equal to thres
[~,Total_Feature]=size(X); % to get the total number of orginal feature
for d = 1: Total_Feature
    if X(d) > thres
%         display (['original X(' num2str(d) ') = ' num2str(X(d))])
        X(d) = 1; 
%         display (['new X(' num2str(d) ') = ' num2str(X(d))])
    else
%         display (['original X(' num2str(d) ') = ' num2str(X(d))])
        X(d) = 0;
%         display (['new X(' num2str(d) ') = ' num2str(X(d))])
    end
end

%% To evaluate the fitness value of each solution after it is converted into binary values
if sum(X == 1) == 0 %if none of the feature is selected (cannot train the classifier then)
  cost = 1; % just assign it with a default value of 1
  error = 1;
else %if there is at least one feature is selected to train classifer
  % identify the index of feature being selected
  s_Feat_index = find(X == 1);
  % identify all selected features based on their indices and store in s_Feat
  % note that only those selected feature will be used to train the classifier
  s_Feat = feat(:, s_Feat_index);
  error = jwrapperKNN(s_Feat,label,HO);
  s_Feat_ratio = length(s_Feat_index)/length(X);
  alpha = 0.99;
  %fitness function to consider both accuracy and size of feature subset
  cost = alpha*error + (1-alpha)*s_Feat_ratio;
  %cost = jwrapperKNN(feat(:, X == 1),label,HO);
end
end


function error = jwrapperKNN(sFeat,label,HO)
%---// Parameter setting for k-value of KNN //
k = 5; 

%% find the indices of training dataset and store them into xtrain & ytrain
TrainSet_Index = find(HO.training == 1);
xtrain = sFeat(TrainSet_Index, :);
ytrain = label(TrainSet_Index, :);

%find the indices of testing dataset and store them into xvalid & yvalid
TestSet_Index = find(HO.test == 1);
xvalid = sFeat(TestSet_Index, :);
yvalid = label(TestSet_Index, :);

% xtrain = sFeat(HO.training == 1,:);
% ytrain = label(HO.training == 1); 
% xvalid = sFeat(HO.test == 1,:); 
% yvalid = label(HO.test == 1); 

%% train the classifer
Model     = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
pred      = predict(Model,xvalid);
num_valid = length(yvalid); 
correct   = 0;
for i = 1:num_valid
  if isequal(yvalid(i),pred(i))
    correct = correct + 1;
  end
end
Acc   = correct / num_valid; 
error = 1 - Acc; 
end


