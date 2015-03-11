function Perf = mlr_test_largescale(L, test_k, Xtrain, Ytrain, Xtest_all, Ytest_all)
% Original code by Brian McFee (brian.mcfee@nyu.edu)
% Modified by Daryl Lim (dklim@ucsd.edu)
% Three major changes from mlr_test:
%
% 1) Distance matrix is computed in batches to conserve memory for large data
% 2) It supports only input given in factored form (i.e L s.t. W = L'L)
% 3) Fixed MAP/NDCG so it returns 0 instead of NaN if there are no relevant
%    points for a given query point.
%
%       L       = m x d matrix such that L'L is the metric of interest
%       test_k  = vector of k-values to use for KNN/Prec@k/NDCG
%       Xtrain  = d-by-n matrix of training data
%       Ytrain  = n-by-1 vector of training labels
%                   OR
%                 n-by-1 cell array where
%                   Y{q,1} contains relevant indices (in 1..n) for point q
%                   OR
%                 n-by-2 cell array where
%                   Y{q,1} contains relevant indices (in 1..n) for point q
%                   Y{q,2} contains irrelevant indices (in 1..n) for point q
%       Xtest   = Test data in training format
%       Ytest   = similar format as Ytrain for test data.
%
%   The output structure Perf contains the mean score for:
%       AUC, KNN, Prec@k, Prec@1, Prec@10, MAP, MRR, NDCG
%



Perf        = struct(                       ...
    'AUC',      [],     ...
    'KNN',      [],     ...
    'PrecAtK',  [],     ...
    'PrecAt1',  [],     ...
    'PrecAt10',  [],     ...
    'MAP',      [],     ...
    'MRR',      [],     ...
    'MFR',      [],     ...
    'NDCG',     [],     ...
    'KNNk',     [],     ...
    'PrecAtKk', [],     ...
    'NDCGk',    []     ...
    );

[d, nTrain, nKernel] = size(Xtrain);
test_k          = min(test_k, nTrain);

batchsize                    = 4000;
nTest_all                    = length(Ytest_all);
startIndices                 = 1:batchsize:nTest_all;
numBatches                   = length(startIndices);
batchWeights                 = zeros(numBatches,1);
batchWeights(1:numBatches-1) = batchsize;
batchWeights(end)            = nTest_all - startIndices(end)+1;
PerfArray = repmat(Perf,numBatches,1);

for batch = 1:numBatches
    startIndex = startIndices(batch);
    endIndex = min(nTest_all,startIndex+batchsize-1);
    tic
    disp(sprintf('Processing indices %d to %d', startIndex, endIndex));
    Xtest = Xtest_all(:,startIndex:endIndex);
    Ytest = Ytest_all(startIndex:endIndex);

    if nargin > 5
        % Knock out the points with no labels
        if ~iscell(Ytest)
            Ibad                = find(isnan(Ytrain));
            Xtrain(:,Ibad,:)    = inf;
        end


    else
        % Leave-one-out validation

        if nargin > 4
            % In this case, Xtest is a subset of training indices to test on
            testRange = Xtest;
        else
            testRange = 1:nTrain;
        end
        Xtest       = Xtrain(:,testRange,:);
        Ytest       = Ytrain(testRange);
    end

    % Build the distance matrix

    [D, I] = mlr_test_distance(L, Xtrain, Xtest);

    if nargin == 5
        % clear out the self-link (distance = 0)
        I       = I(2:end,:);
        D       = D(2:end,:);
    end


    nTest       = length(Ytest);

    % Compute label agreement
    if ~iscell(Ytest)
        % First, knock out the points with no label
        Labels  = Ytrain(I);
	Agree   = bsxfun(@eq, Ytest', Labels);
        % We only compute KNN error if Y are labels
        [PerfArray(batch).KNN, PerfArray(batch).KNNk] = mlr_test_knn(Labels, Ytest, test_k);
    else
        if nargin > 5
            Agree   = zeros(nTrain, nTest);
        else
            Agree   = zeros(nTrain-1, nTest);
        end
        for i = 1:nTest
            Agree(:,i) = ismember(I(:,i), Ytest{i,1});
        end

        Agree = reduceAgreement(Agree);
    end

    % Compute AUC score
    PerfArray(batch).AUC    = mlr_test_auc(Agree);

    % Compute MAP score
    PerfArray(batch).MAP    = mlr_test_map(Agree);

    % Compute MRR score
    PerfArray(batch).MRR    = mlr_test_mrr(Agree);
    PerfArray(batch).MFR    = mlr_test_mfr(Agree);

    % Compute prec@k
    [PerfArray(batch).PrecAtK, PerfArray(batch).PrecAtKk] = mlr_test_preck(Agree, test_k);
    [PerfArray(batch).PrecAt1, ~] = mlr_test_preck(Agree, 1);
    [PerfArray(batch).PrecAt10, ~] = mlr_test_preck(Agree, 10);

    % Compute NDCG score
    [PerfArray(batch).NDCG, PerfArray(batch).NDCGk] = mlr_test_ndcg(Agree, test_k);
    toc
end

SNames = fieldnames(Perf);
for loopIndex = 1:numel(SNames)
    if ~isempty([PerfArray.(SNames{loopIndex})])
        Perf.(SNames{loopIndex}) = ([PerfArray.(SNames{loopIndex})]*batchWeights)/sum(batchWeights);
    end
end

end


function [D,I] = mlr_test_distance(L, Xtrain, Xtest)

% CASES:
%   Raw:                        L = []
%   Low rank                    L = d-by-m
%   Linear, diagonal:           L = d-by-1

[d, nTrain, nKernel] = size(Xtrain);
nTest = size(Xtest, 2);

if isempty(L)
    % L = []  => native euclidean distances
    D = (bsxfun(@plus, dot(Xtest,Xtest)', dot(Xtrain,Xtrain)) - 2*Xtest'*Xtrain)';


elseif size(L,2) == d && size(L,1) < d
    % Low rank L!

    Xtrt = L * Xtrain;
    Xtet = L * Xtest;

    D = (bsxfun(@plus, dot(Xtet,Xtet)', dot(Xtrt,Xtrt)) - 2 * Xtet'*Xtrt)';

elseif size(L,1) == d && size(L,2) == 1
    %diagonal
    Xtrt = bsxfun(@times,Xtrain,L);
    Xtet = bsxfun(@times,Xtest,L);
    D = (bsxfun(@plus, dot(Xtrain,Xtrt)', dot(Xtest,Xtet)) - 2 * Xtrain'*Xtet);

elseif size(L,1) == d && size(L,2) == d
    disp('treating matrix as factored form L')
    Xtrt = L * Xtrain;
    Xtet = L * Xtest;

    D = (bsxfun(@plus, dot(Xtet,Xtet)', dot(Xtrt,Xtrt)) - 2 * Xtet'*Xtrt)';
else
    % Error?
    error('Cannot determine metric mode.');

end

[v,I]   = sort(D, 1);
end

function [ndcg, ndcgk] = mlr_test_ndcg(Agree_tags,test_k);
trunc = 1;
nTrain = size(Agree_tags,1);

topk_scores = Agree_tags(1:test_k,:);
disc = 1./[log2(2:nTrain+1)];
if trunc == 1
    sort_norel = sort(Agree_tags,1, 'descend');
else
    sort_norel = sort(topk_scores,1, 'descend');
end

dcg = disc(1:test_k)*topk_scores;
nor = disc(1:test_k)*sort_norel(1:test_k,:);
dcgall = disc * Agree_tags;
norall = disc * sort_norel;

ndcgk = mean(dcg./(nor+eps));
ndcg = mean(dcgall./(norall+eps));
end


function [PrecAtK, PrecAtKk] = mlr_test_preck(Agree, test_k)

PrecAtK        = -Inf;
PrecAtKk       = 0;
for k = test_k
    b   = mean( mean( Agree(1:k, :), 1 ) );
    if b > PrecAtK
        PrecAtK = b;
        PrecAtKk = k;
    end
end
end

function [KNN, KNNk] = mlr_test_knn(Labels, Ytest, test_k)

KNN        = -Inf;
KNNk       = 0;
for k = test_k
    % FIXME:  2012-02-07 16:51:59 by Brian McFee <bmcfee@cs.ucsd.edu>
    %   fix these to discount nans

    b   = mean( mode( Labels(1:k,:), 1 ) == Ytest');
    if b > KNN
        KNN    = b;
        KNNk   = k;
    end
end
end

function MAP = mlr_test_map(Agree);

nTrain      = size(Agree, 1);
MAP         = bsxfun(@ldivide, (1:nTrain)', cumsum(Agree, 1));
MAP         = mean(sum(MAP .* Agree, 1)./ (sum(Agree, 1)+eps));
end

function MRR = mlr_test_mrr(Agree);

nTest = size(Agree, 2);
MRR        = 0;
for i = 1:nTest
    MRR    = MRR  + (1 / find(Agree(:,i), 1));
end
MRR        = MRR / nTest;
end

function MFR = mlr_test_mfr(Agree);

nTest = size(Agree, 2);
MFR        = 0;
for i = 1:nTest
    MFR    = MFR  + (find(Agree(:,i), 1));
end
MFR        = MFR / nTest;
end

function AUC = mlr_test_auc(Agree)

TPR             = cumsum(Agree,     1);
FPR             = cumsum(~Agree,    1);

numPos          = TPR(end,:);
numNeg          = FPR(end,:);

TPR             = mean(bsxfun(@rdivide, TPR, numPos+eps),2);
FPR             = mean(bsxfun(@rdivide, FPR, numNeg+eps),2);
AUC             = diff([0 FPR']) * TPR;
end

function A = reduceAgreement(Agree)
nPos = sum(Agree,1);
nNeg = sum(~Agree,1);

goodI = find(nPos > 0 & nNeg > 0);
A = Agree(:,goodI);
end
