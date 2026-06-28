function [cost, error] = jFitnessFunction(feat, label, X, HO, thres)
% jFitnessFunction
% -------------------------------------------------------------------------
% Wrapper fitness for binary feature selection.
%
% IMPORTANT (R3-1, data leakage): the data passed in here (feat, label) is
% the TRAINING data only, and HO is an INNER validation partition built
% solely from that training data (see buildInnerPartition in Main.m). The
% held-out TEST set is never visible to this function, so it cannot leak
% into the search. Final test-set accuracy is computed once, separately,
% by jFinalEval in Main.m.
%
% Inputs
%   feat   : training feature matrix (instances x features)
%   label  : training label vector  (instances x 1)
%   X      : real-valued position vector (1 x features)
%   HO     : inner cvpartition (hold-out OR k-fold; see jwrapperKNN)
%   thres  : binary mapping threshold (X(d) > thres -> feature selected)
%
% Outputs
%   cost   : alpha*error + (1-alpha)*selection_ratio  (minimised)
%   error  : inner-validation classification error of the selected subset
% -------------------------------------------------------------------------

% --- Binary mapping: select feature d if X(d) > thres ---
nFeat = numel(X);
sel   = X > thres;                 % logical selection mask (vectorised)

if ~any(sel)                       % no feature selected -> classifier cannot train
    cost  = 1;                     % worst-case penalty
    error = 1;
    return;
end

% --- Evaluate the selected subset on the inner validation partition ---
s_Feat_index = find(sel);
s_Feat       = feat(:, s_Feat_index);
error        = jwrapperKNN(s_Feat, label, HO);

% --- Combined objective: accuracy vs subset size ---
s_Feat_ratio = numel(s_Feat_index) / nFeat;
alpha        = 0.99;               % accuracy weight (R2-4: sensitivity sweep point)
cost         = alpha * error + (1 - alpha) * s_Feat_ratio;
end


function error = jwrapperKNN(sFeat, label, HO)
% jwrapperKNN
% -------------------------------------------------------------------------
% KNN classification error on the INNER validation partition.
%
% Supports BOTH inner schemes returned by buildInnerPartition:
%   - single stratified hold-out : HO.NumTestSets == 1
%   - stratified k-fold          : HO.NumTestSets  > 1  (error averaged
%                                  across folds, R3-2 small-sample stability)
%
% Leakage-free: HO is built only from training data; the outer test set is
% never passed in.
% -------------------------------------------------------------------------

k = 5;                             % KNN neighbours (R2-4: single point of control)

nSets   = HO.NumTestSets;          % 1 for hold-out, >1 for k-fold
foldErr = zeros(nSets, 1);

for f = 1:nSets
    trMask = training(HO, f);
    teMask = test(HO, f);

    xtrain = sFeat(trMask, :);
    ytrain = label(trMask, :);
    xvalid = sFeat(teMask, :);
    yvalid = label(teMask, :);

    Model = fitcknn(xtrain, ytrain, 'NumNeighbors', k);
    pred  = predict(Model, xvalid);

    foldErr(f) = 1 - (sum(yvalid == pred) / numel(yvalid));   % vectorised acc
end

error = mean(foldErr);             % single value for hold-out; mean for k-fold
end