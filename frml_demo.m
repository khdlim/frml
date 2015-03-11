load IN300_folds.mat

Xtrain = folds(1).Ktrain;
Ytrain = folds(1).Ytrain;
Xval = folds(1).Kval;
Yval = folds(1).Yval;
Xtest = folds(1).Ktest;
Ytest = folds(1).Ytest;

dr = 30;
[d,N] = size(Xtrain);
L = randn(d,dr);
%for each query, use every same-class point as positive point
indiv_similarity = 0;
target_k = N;
[similar,different] = get_sim_diff(indiv_similarity,target_k, Xtrain,Ytrain);

%% - Code broken
% target_k = 3;
% [similar, different] =  get_sim_diff(1,target_k, Xtrain,Ytrain);
% Ytrain = cell(length(Ytrain),2)
% for idx = 1:length(Ytrain)
%     Ytrain{idx,1} = similar{idx}
%     Ytrain{idx,2} = different{idx}
% end

%set parameters for experiment. commented parameters are not necessary as
%they are the default values.

params.lam = 1e-3;
params.dr = dr;
params.num_iter = 100000;
params.n0 = 4096;
% params.n1 = 0;
% params.valid_criteria = 'MAP';
params.report_interval = 10000;
% params.regularizer = 'lmnn';
params.dr = 30;
params.manifold = 1;
% params.mode = 'warp';
% params.loss = 'rec';
params.verbose = 1;
params.rank_thresh = 0.1;
params.test_k = 3;


[L_new, Diag, conv] = frml_warp(L, Xtrain,Ytrain, Xval, Yval, params, similar, different);
mlr_test_largescale(L_new', params.test_k, Xtrain, Ytrain, Xtest, Ytest)
