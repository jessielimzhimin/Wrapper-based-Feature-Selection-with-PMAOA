% -------------------------------------------------------------------%
% Wrapper-based feature selection method demo version                 %
%-------------------------------------------------------------------%

%---Input------------------------------------------------------------
% feat     3 feature vector (instances x features)
% label    : label vector (instances x 1)
% N        : Number of solutions
% max_Iter : Maximum number of iterations
% CR       : Crossover rate

%---Output-----------------------------------------------------------
% sFeat    : Selected features (instances x features)
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%--------------------------------------------------------------------

clc
clear
close all
disp('Loading DS 3 - Dermatology Data Set')
fileID = fopen('dermatology.data');
format = repmat('%f', [1, 35]);
C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
fclose(fileID);
feat = C{1,1}(:, 1:end-1);
label = C{1,1}(:, end);
% Set 20% data as validation set
ho = 0.2;
% Hold-out method
HO = cvpartition(label,'HoldOut',ho,'Stratify',false);
% Parameter setting
N = 20;         % Population size (20)
max_FEs = 2000; % Maximum fitness evaluation number (2000)
CR = 0.9;       % Crossover rate (0.9)
max_Run = 20;   % Maximum simulation run (30 Runs)
Algorithm_Name = 'PMAOA';
for r = 1: max_Run

    tic
    [sFeat,Sf,Nf,curve, ErrorCurve] = jPMAOA(feat,label,N,max_FEs,HO);
    disp(['Run = ' num2str(r) ', best (' Algorithm_Name ') = ' num2str(curve(end))])
    Time_Per_Run(r)=toc;
    Nf_All(r,1) = Nf;
    curve_All(r,:) = curve(end-max_FEs+1:end);
    ErrorCurve_All(r,:) = ErrorCurve(end-max_FEs+1:end);
    Sf_All{r} = Sf;
    [~, total_feat]=size(feat);
    alpha = 0.99;
    FeatureSubsetSize_All(r,:)= floor(total_feat*(curve_All(r,:)-alpha*ErrorCurve_All(r,:))/(1-alpha));
    for k = 1:length(curve)
        raw = (curve(k) - alpha*ErrorCurve(k))/(1-alpha);
        if raw < 0
            fprintf('k=%d: F=%.4f,  E=%.4f,  raw=%.4f\n', k, curve(k), ErrorCurve(k), raw);
            break;
        end
    end
end

%% Post analysis based on results obtained
% (1) Mean classification error
Mean_Error = mean(ErrorCurve_All(:,end));
% (2) Standard deviation of classification error
StdDev_Error = std(ErrorCurve_All(:,end));
% (3) Mean classification accuracy
Mean_Acc = mean(1-ErrorCurve_All(:,end));
% (4) Standard deviation of classification accurary
StdDev_Acc = std(1-ErrorCurve_All(:,end));
% (5) Average number of selected features
Ave_Nf = mean(Nf_All);
% (6) Ratio of feature being selected
Ratio_NF = Ave_Nf/size(feat,2);
% (7) Average time to complete
Ave_Time = mean(Time_Per_Run);
% (7) Average fitness
All_Fitness = curve_All(:,end);
Ave_Fitness = mean(All_Fitness);
clc
disp(['Summary of results by ' Algorithm_Name ' for Dermatology Dataset'])
disp(['Mean error = '  num2str(Mean_Error)])
disp(['StdDev_Error = '  num2str(StdDev_Error)])
disp(['Mean_Acc = '  num2str(Mean_Acc)])
disp(['StdDev_Acc = '  num2str(StdDev_Acc)])
disp(['Average Selected Features = '  num2str(Ave_Nf)])
disp(['Total Features = '  num2str(size(feat,2))])
disp(['Ratio_Nf = '  num2str(Ratio_NF)])
disp(['Average Time = '  num2str(Ave_Time)])
disp(['Average Fitness = '  num2str(Ave_Fitness)])