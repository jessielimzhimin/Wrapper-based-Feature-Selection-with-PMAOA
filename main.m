% -------------------------------------------------------------------%
% PMAOA Wrapper-Based Feature Selection: Main Script                  %
%-------------------------------------------------------------------%
% Runs the proposed PMAOA across 20 UCI benchmark datasets using a
% leakage-free nested hold-out protocol: the outer test split is held out
% from the optimiser entirely, and the inner split is used only by the
% wrapper fitness function to guide the feature selection search.

%
%---Input------------------------------------------------------------
% feat     : feature matrix (instances x features)
% label    : label vector (instances x 1)
% N        : Number of solutions
% max_FEs  : Maximum number of fitness evaluations
%
%---Output-----------------------------------------------------------
% sFeat    : Selected features (instances x features)
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%--------------------------------------------------------------------

clc
clear
close all

%% ============================ DATASETS ===================================
% The 20 UCI benchmark datasets used in this study (database indices, as
% defined in Load_UCI_Data.m).
DatasetList = 1:20;

Algorithm_Name = 'PMAOA';

% Path to the Dataset and Algorithm folders (adjust to your local
% directory structure).
addpath('Dataset\')
addpath(['Algorithm\' Algorithm_Name])

%% ============================ PARAMETERS =================================
ho_outer = 0.2;   % fraction held out as the final TEST set
ho_inner = 0.2;   % fraction of TRAINING used as inner validation (fitness)

N       = 20;     % Population size
max_FEs = 10;   % Maximum fitness evaluation number
max_Run = 1;     % Number of independent runs per dataset

outdir = 'Data Result_PMAOA';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
figdir = 'Figure Result';
if ~exist(figdir, 'dir')
    mkdir(figdir);
end

%% ======================== MAIN COMPARISON LOOP ============================
for di = 1:numel(DatasetList)
    DataSet_Index = DatasetList(di);
    [feat, label] = Load_UCI_Data(DataSet_Index);

    % Untouched master copy. The held-out TEST set is carved from this
    % each run and is NEVER seen by the optimiser (leakage-free).
    feat_full  = feat;
    label_full = label;

    % Confusion-matrix store (for F1 / recall / precision / MCC).
    ConfMat_All = cell(max_Run,1);

    % Per-run train/test class distributions, for reporting.
    ClassDist_All = cell(max_Run,1);

    % Which inner-validation scheme each run used (hold-out vs k-fold).
    InnerScheme_All = cell(max_Run,1);

    % Preallocate
    Time_Per_Run          = zeros(max_Run,1);
    Nf_All                = zeros(max_Run,1);
    curve_All              = zeros(max_Run,max_FEs);
    ErrorCurve_All          = zeros(max_Run,max_FEs);
    FeatureSubsetSize_All   = zeros(max_Run,max_FEs);
    Sf_All                  = cell(max_Run,1);
    TestAcc_All             = zeros(max_Run,1);

    for r = 1:max_Run

        tic
        % -----------------------------------------------------------------
        % Leakage-free, nested, stratified, per-run resampling:
        %   OUTER -> held-out TEST set; never enters the fitness function
        %   INNER -> validation split drawn ONLY from the training portion;
        %            this is what the wrapper KNN scores during the search
        % rng(r) makes each run's split reproducible. The same seeding
        % convention is applied identically when evaluating every
        % competing algorithm reported in the manuscript, so the paired
        % Wilcoxon/Friedman tests remain valid across all comparisons.
        % -----------------------------------------------------------------
        rng(r);

        outer   = cvpartition(label_full, 'HoldOut', ho_outer, 'Stratify', true);
        trIdx   = training(outer);
        teIdx   = test(outer);

        feat    = feat_full(trIdx, :);    % TRAINING data only -> to optimiser
        label   = label_full(trIdx, :);
        featTe  = feat_full(teIdx, :);    % TEST data -> held out until final eval
        labelTe = label_full(teIdx, :);

        % Inner validation split, drawn from the TRAINING data only.
        [HO, InnerScheme_All{r}] = buildInnerPartition(label, ho_inner);

        % Record class distributions for reporting.
        ClassDist_All{r}.train = countClasses(label);
        ClassDist_All{r}.test  = countClasses(labelTe);

        % ---- Run PMAOA ----
        [sFeat, Sf, Nf, curve, ErrorCurve] = jPMAOA(feat, label, N, max_FEs, HO);

        disp(['Run = ' num2str(r) ', best (' Algorithm_Name ') = ' num2str(curve(end))])

        Time_Per_Run(r) = toc;
        Nf_All(r,1)     = Nf;
        curve_All(r,:)       = curve(end-max_FEs+1:end);
        ErrorCurve_All(r,:)  = ErrorCurve(end-max_FEs+1:end);
        Sf_All{r}            = Sf;

        % Final evaluation on the HELD-OUT TEST set: refit KNN (k=5) on the
        % FULL training data with the selected features, then score ONCE
        % on the untouched test set. This is the number reported in the
        % comparison tables.
        [ConfMat_All{r}, TestAcc_All(r), ClassOrder] = ...
            jFinalEval(feat, label, featTe, labelTe, Sf);

        % Feature-subset-size trajectory, recalculated from fitness/error.
        [~, total_feat] = size(feat);
        alpha = 0.99;
        FeatureSubsetSize_All(r,:) = floor(total_feat * ...
            (curve_All(r,:) - alpha*ErrorCurve_All(r,:)) / (1-alpha));
    end

    %% ---- post-analysis ----
    % NOTE: the curve-based metrics below describe the INNER VALIDATION
    % trajectory only (search behaviour / convergence). They are NOT the
    % reported performance. The headline numbers are the held-out TEST
    % metrics computed further down (Mean_TestAcc etc.).
    Mean_Error   = mean(ErrorCurve_All(:,end));
    StdDev_Error = std(ErrorCurve_All(:,end));
    Mean_Acc     = mean(1 - ErrorCurve_All(:,end));
    StdDev_Acc   = std(1 - ErrorCurve_All(:,end));
    Ave_Nf       = mean(Nf_All);
    Ratio_NF     = Ave_Nf / size(feat_full,2);
    Ave_Time     = mean(Time_Per_Run);
    All_Fitness  = curve_All(:,end);
    Ave_Fitness  = mean(All_Fitness);

    % HEADLINE metrics: HELD-OUT TEST set (leakage-free).
    Mean_TestAcc   = mean(TestAcc_All);
    StdDev_TestAcc = std(TestAcc_All);
    Mean_TestError = 1 - Mean_TestAcc;

    % Best run chosen by INNER validation error (no peeking at test set).
    [~, best_run] = min(ErrorCurve_All(:,end));
    Best_ConfMat  = ConfMat_All{best_run};
    Best_TestAcc  = TestAcc_All(best_run);

    fprintf('\nSummary of results by %s for Dataset = %d\n', Algorithm_Name, DataSet_Index);
    fprintf('Mean TEST accuracy (reported) = %.4f\n', Mean_TestAcc);
    fprintf('StdDev TEST accuracy          = %.4f\n', StdDev_TestAcc);
    fprintf('Average Selected Features     = %.2f / %d\n', Ave_Nf, size(feat_full,2));
    fprintf('Average Time                  = %.2f s\n', Ave_Time);

    %% ---- figures ----
    figure('Visible','off');
    plot(1:max_FEs, mean(curve_All)); xlabel('Number of Fitness Evaluation');
    ylabel('Fitness Value'); title(Algorithm_Name); grid on;
    saveas(gcf, fullfile(figdir, [Algorithm_Name '_Convergence_DS_' num2str(DataSet_Index)]), 'tiff');
    close(gcf);

    figure('Visible','off');
    Plot_Accuracy_vs_FeatureSubsetSize(ErrorCurve_All, FeatureSubsetSize_All);
    saveas(gcf, fullfile(figdir, [Algorithm_Name '_AccuracyvsFeatureSubsetSize_DS_' num2str(DataSet_Index)]), 'tiff');
    close(gcf);

    figure('Visible','off');
    confusionchart(Best_ConfMat, ClassOrder, ...
        'Title', [Algorithm_Name ' - Best Run Confusion Matrix (DS ' num2str(DataSet_Index) ')'], ...
        'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
    saveas(gcf, fullfile(figdir, [Algorithm_Name '_ConfusionMatrix_DS_' num2str(DataSet_Index)]), 'tiff');
    close(gcf);

    %% ---- save results ----
    filename = [Algorithm_Name '_Result_DS_' num2str(DataSet_Index) '.mat'];
    filelocation = fullfile(outdir, filename);
    save(filelocation, 'Nf_All', 'curve_All', 'Sf_All', 'ErrorCurve_All', ...
        'Mean_Error', 'StdDev_Error', 'Mean_Acc', 'StdDev_Acc', 'Ave_Nf', ...
        'Ratio_NF', 'Ave_Time', 'All_Fitness', 'Ave_Fitness', 'N', ...
        'max_FEs', 'max_Run', 'FeatureSubsetSize_All', ...
        'ConfMat_All', 'TestAcc_All', 'ClassOrder', ...
        'Best_ConfMat', 'Best_TestAcc', 'best_run', ...
        'Mean_TestAcc', 'StdDev_TestAcc', 'Mean_TestError', ...
        'ClassDist_All', 'InnerScheme_All', 'ho_outer', 'ho_inner');

    clearvars -except di DatasetList Algorithm_Name N max_FEs max_Run ...
        ho_outer ho_inner outdir figdir
end   % <-- dataset loop (for di)


%% ======================= LOCAL FUNCTIONS ================================

function [confMat, testAcc, classOrder] = jFinalEval(featTr, labelTr, featTe, labelTe, Sf)
% Leakage-free final evaluation on the held-out TEST set.
% Trains KNN (k = 5, matching the wrapper fitness) on the FULL training
% data using the selected features, then scores ONCE on the test set.
%   confMat    : rows = true class, cols = predicted class
%   testAcc    : held-out test-set classification accuracy
%   classOrder : fixed class list (same size matrix for every run)

classOrder = unique([labelTr; labelTe]);   % full, fixed class list

if isempty(Sf)               % guard: no features selected
    confMat = zeros(numel(classOrder));
    testAcc = NaN;
    return;
end

Model = fitcknn(featTr(:, Sf), labelTr, 'NumNeighbors', 5);
pred  = predict(Model, featTe(:, Sf));

% Force fixed ordering so every run returns a same-size N x N matrix
confMat = confusionmat(labelTe, pred, 'Order', classOrder);
testAcc = sum(pred == labelTe) / numel(labelTe);
end


function tbl = countClasses(y)
% Class distribution as a [classLabel, count] matrix.
c = unique(y);
n = zeros(numel(c), 1);
for ii = 1:numel(c)
    n(ii) = sum(y == c(ii));
end
tbl = [c(:), n];
end


function [innerPart, scheme] = buildInnerPartition(ytr, ho_inner)
% Leakage-free inner validation partition, built ONLY from training labels.
% A single 80/20 hold-out is unreliable on small/imbalanced data, so
% datasets whose smallest class is small automatically switch to stratified
% k-fold (averaged inside the wrapper). Larger datasets keep a single
% stratified hold-out for speed. Returned partition works with jwrapperKNN
% for both schemes via HO.NumTestSets.
classes  = unique(ytr);
counts   = arrayfun(@(c) sum(ytr == c), classes);
minClass = min(counts);

if minClass >= 5
    innerPart = cvpartition(ytr, 'HoldOut', ho_inner, 'Stratify', true);
    scheme    = 'stratified holdout';
elseif minClass >= 2
    nFolds    = min(5, minClass);          % each class must fill every fold
    innerPart = cvpartition(ytr, 'KFold', nFolds, 'Stratify', true);
    scheme    = sprintf('stratified %d-fold', nFolds);
else
    % Degenerate: a class with a single training sample cannot be
    % stratified at all. Fall back to a plain hold-out and warn so the
    % situation is visible rather than silently mishandled.
    warning('buildInnerPartition:tinyClass', ...
        'A class has <2 training samples; using non-stratified hold-out.');
    innerPart = cvpartition(numel(ytr), 'HoldOut', ho_inner);
    scheme    = 'non-stratified holdout (degenerate)';
end
end

function Plot_Accuracy_vs_FeatureSubsetSize(ErrorCurve_All, FeatureSubsetSize_All)
    accData = 1 - mean(ErrorCurve_All);                   % Accuracy = 1 - mean(Error)
    fssData = floor(mean(FeatureSubsetSize_All));         % Feature Subset Size (rounded down)
    x = 1:length(accData);

    % Compute min, max, and padded limits for Accuracy
    accMin = min(accData);
    accMax = max(accData);
    accRange = accMax - accMin;

    if accRange == 0
        accRange = 0.1;
    end

    paddingRatio = 0.05;  % 5% padding
    accLower = accMin - paddingRatio * accRange;
    accUpper = accMax + paddingRatio * accRange;

    % Compute min, max, and padded limits for Feature Subset Size
    fssMin = min(fssData);
    fssMax = max(fssData);
    fssRange = fssMax - fssMin;

    if fssRange == 0
        fssRange = 0.1;
    end
    fssLower = fssMin - paddingRatio * fssRange;
    fssUpper = fssMax + paddingRatio * fssRange;

    % Plot Accuracy on the Left y-axis
    yyaxis left
    plot(x, accData, '-b', 'LineWidth', 2);
    ylabel('Accuracy');
    ylim([accLower, accUpper]);  % Apply padded limits for Accuracy

    % Plot Feature Subset Size on the Right y-axis
    yyaxis right
    plot(x, fssData, '--r', 'LineWidth', 2);
    ylabel('Feature Subset Size');
    ylim([fssLower, fssUpper]);  % Apply padded limits for Feature Subset Size

    % Common x-axis and other plot settings
    xlabel('Fitness Evaluation Number');
    grid on;
    legend('Accuracy','Feature Subset Size','Location','southeast');

    % Optionally add a title if needed:
    % title('Variation of Accuracy and Feature Subset Size with Fitness Evaluations');
end
