function [L, Diag, converged] = frml_warp(init_L, Xtrain,Ytrain, Xval, Yval, params, sim, dif)

% d = number of features per training point
% N = number of training points
%
% init_L: initial value of L such that LL' = W.
%
% Xtrain: d x N matrix of training points
%
% Ytrain: 3 options
% --Option 1 (Classification): N x 1 vector, where Ytrain(i) = class of Xtrain(:,i), from 1..k for k classes.
% --Option 2 (Ranking)       : N x 1 cell, where each cell entry Ytrain{i} is a vector containing indices of X
% --Option 3 (Ranking)       : N x 2 cell, where each Ytrain{i,1} is a vector containing
%                              indices of Xtrain which are similar to training point i and
%                              each Ytrain{i,2} is a vector containing indices of Xtrain(:,i) which are the
%                              dissimilar points to training point i
%
% Example: x1, x2 = class1, x3, x4, x5 = class2
% Option 1: Ytrain = [1,1,2,2,2]
% Option 2:  Ytrain{1} = [2]
%            Ytrain{2} = [1],
%            Ytrain{3} = [4,5]
%            Ytrain{4} = [3,5]
%            Ytrain{5} = [3,4]
% Option 3:  Ytrain{1,1} = [2], Ytrain{1,2} = [3,4,5]
%            Ytrain{2,1} = [1], Ytrain{1,2} = [3,4,5]
%            ... etc
% For ranking problems, Option 1 is not possible.
%
% Xval, Yval: Similar format for Xtrain/Ytrain. Validation set used to measure training progress.
%
% params: Struct of optional parameters. All are optional with a default value
% except params.dr which is required.
%
% Their use is as follows:
% dr          : rank of low-rank metric W. Equivalently, target reduced
%               dimension
% lam         : regularizer weight. Default 0.01
% cutoff_k    : cutoff for 'pak' and 'trec' warp losses (see loss) below. Default 3
% test_k	    : number of neighbours k for measuring kNN performance on validation set (if applicable). Default 3
% num_iter    : number of training samples to see (default 10000). Using minibatches counts as observable iterations,
%               so minibatch of 10 samples means 1000 minibatches will be seen with num_iter = 10000.
% n0	    : stepsize parameter 1. stepsize = n0/(1+n1*iteration_number)
% n1	    : stepsize parameter 2.
% mode	    : 'warp' or 'auc'. 'warp' uses warp loss sampling while 'auc' only samples a single negative point per query.
% loss	    : 'pak': sets alpha_i = 1 for i < cutoff_k, and zero otherwise
% 	     'trec': sets alpha_i = 1/i for i < cutoff_k, and zero otherwise
% 	      'auc': sets alpha_i = 1/M, where M is the number of negative points.
% 	      'rec': sets alpha_i = 1/i
% 	      Only applicable in 'warp' mode
%
% sparse_matrix: Use if training matrix Xtrain is sparse.
%
% valid_criteria: Which performance measure to use on the validation set.
%
% batchsize: minibatch size. Default 5
%
% manifold: If 1, use Riemannian manifold gradient descent on LL'. If 0, use regular gradient descent on L.
%
% report_interval: How often validation set performance is measured.
%
% verbose: if 1, prints out a list of all actual parameter settings (including defaults) prior to running
%
% iter_offset: offsets the iteration number reported every <report_interval>
%              iterations in the variable 'Diag' by iter_offset.
% time_offset: offsets the starting time reported every report_interval in the
%              variable 'Diag' by time_offset.
% rank_thresh: Controls the maximum WARP samples.
%              For a desired gamma value in the ICML paper, rank_thresh
%              should be set to 1/gamma.
% regularizer: 'lmnn' is the regularizer in the paper, 'l2' is the squared frobenius norm.
%
% sim,dif: Precalculated indices of similar/dissimilar points to save computation time.
% If Ytrain = Option 1, then both sim and dif should be Kx1 cell arrays.
%             sim{i} should be a vector which contains all indices of points
%             belonging to class i, while dif(i) contains all indices of points
%             not belonging to class i.
%             e.g. Ytrain = [1,2,3,2,3,1]
%             sim{1} = [1,6]
%             dif{1} = [2,3,4,5]
%             sim{2} = [2,4]
%             dif{2} = [1,3,5,6]
%             sim{3} = [3,5]
%             dif{3} = [1,2,4,6]
% If Ytrain = Option 2, then sim is not used, dif should be a Nx1 cell array
%             where dif{i} = setdiff([1:N]',Ytrain{i,1}
% If Ytrain = Option 3, then sim/dif not used
%
% The provided function "get_sim_diff.m" can be used to generate {sim. dif}
% Ytrain as input
%
%

% params              = struct(                           ...
%                             'lam',              [],     ...
%                             'dr',               [],     ...
%                             'num_iter',         [],     ...
%                             'mode',             [],     ...
%                             'cutoff_k',         [],     ...
%                             'test_k',           [],     ...
%                             'n0',               [],     ...
%                             'n1',               [],     ...
%                             'loss',             [],     ...
%                             'burn_in',          [],     ...
%                             'warm_restart',     [],     ...
%                             'init_L',           [],     ...
%                             'sparse',           [],     ...
%                             'batchsize',        [],     ...
%                             'manifold',         [],     ...
%                             'valid_criteria',   [],      ...
%                             'verbose',          [],      ...
%                             'iter_offset',      [],      ...
%                             'time_offset',      [],      ...
%                             'rank_thresh',      [],      ...
%                             'regularizer',      [],      ...
%                             'report_interval',  []      ...
%                             );

runtime_params.lam             = set_default(params, 'lam',               .01);
runtime_params.cutoff_k        = set_default(params, 'cutoff_k',          3);
runtime_params.test_k 	       = set_default(params, 'test_k',        	   3);
runtime_params.num_iter        = set_default(params, 'num_iter',          10000);
runtime_params.n0              = set_default(params, 'n0',                .1);
runtime_params.n1              = set_default(params, 'n1',                0);
runtime_params.mode            = set_default(params, 'mode',              'warp');
runtime_params.loss            = set_default(params, 'loss',              'rec');
runtime_params.sparse_matrix   = set_default(params, 'sparse_matrix',     0);
runtime_params.valid_criteria  = set_default(params, 'valid_criteria',    'MAP');
runtime_params.batchsize       = set_default(params, 'batchsize',         5);
runtime_params.manifold        = set_default(params, 'manifold',          1);
runtime_params.report_interval = set_default(params, 'report_interval',   1000);
runtime_params.iter_offset     = set_default(params, 'iter_offset',       0);
runtime_params.time_offset     = set_default(params, 'time_offset',       0);
runtime_params.rank_thresh     = set_default(params, 'rank_thresh',       1);
runtime_params.regularizer     = set_default(params, 'regularizer',       'lmnn');
runtime_params.verbose         = set_default(params, 'verbose',           0);
runtime_params.dr              = params.dr;

lam             = runtime_params.lam;
cutoff_k        = runtime_params.cutoff_k;
test_k 	        = runtime_params.test_k;
num_iter        = runtime_params.num_iter;
n0              = runtime_params.n0;
n1              = runtime_params.n1;
mode            = runtime_params.mode;
loss            = runtime_params.loss;
sparse_matrix   = runtime_params.sparse_matrix;
valid_criteria  = runtime_params.valid_criteria;
batchsize       = runtime_params.batchsize;
manifold        = runtime_params.manifold;
report_interval = runtime_params.report_interval;
time_offset     = runtime_params.time_offset;
iter_offset     = runtime_params.iter_offset;
rank_thresh     = runtime_params.rank_thresh;
regularizer     = runtime_params.regularizer;
verbose         = runtime_params.verbose;
dr              = runtime_params.dr;

% rand('seed',0);

if verbose
    disp(runtime_params)
end

if sparse_matrix == 1
    disp('Using sparse matrix')
end

[dim,n]         = size(Xtrain);
L               = init_L;
if ~isequal(size(L),[dim,dr])
    error('Dimension mismatch, abort')
end

% generate WARP loss tables
if strcmp(loss, 'pak')
    loss_table = [0 ones(1,cutoff_k) zeros(1,n-cutoff_k)];
elseif strcmp(loss, 'rec')
    loss_table = [0 1./[1:n]];
elseif strcmp(loss, 'ndcg')
    loss_table = 1-1./(log2([2:(n+2)]));
    loss_table = [0 diff(loss_table)];
elseif strcmp(loss, 'trec')
    loss_table = [0 1./[1:cutoff_k] zeros(1,n-cutoff_k)];
elseif strcmp(loss ,'auc')
    loss_table = [0 1/n* ones(1,n)];
elseif isempty(loss)
    loss_table = [0 1./[1:n]];
else
    disp('Invalid loss')
end
loss_table = cumsum(loss_table);

% set regularizer
switch regularizer
    case char('lmnn')
        reg_maxindex = 1;
    case char('l2')
        reg_maxindex = dr;
    otherwise
        error('Invalid regularizer')
end
reg_indices = randperm(reg_maxindex);
reg_counter = 1;

% set indiv_similarity
if ~iscell(Ytrain) && length(sim) == length(unique(Ytrain))
    indiv_similarity = 0;
else
    indiv_similarity = 1;
end

% variables for diagnostics
ns_all          = zeros(1,num_iter);
time_all        = [];
iter_count      = [];
allPerf         = struct;

% variables for cross validation/early stopping
dip_ctr         = 0;
bestL           = init_L;
best_perf       = 0;
converged       = 0;
tol             = 0;  
dip_thresh      = 50;

% variables for gradient update
minibatch_count = 0;
grad_rank       = 0;
avg_grad        = zeros(size(L));
p               = zeros(dim, batchsize*3);
q               = zeros(dim, batchsize*3);


if manifold
    %initialize pseudoinverse of L
    pL = (L'*L)\L';
end

% start timer
t = tic;

counter     = 1;
indices     = randperm(n);
for idx = 1:num_iter
    stepsize = n0/(1+n1*idx);
    if mod(idx,report_interval) == 0

        perf = mlr_test_largescale(L', test_k, Xtrain, Ytrain, Xval, Yval);

        SNames = fieldnames(perf);
        for loopIndex = 1:numel(SNames)
            if ~isfield(allPerf, SNames{loopIndex})
                allPerf = setfield(allPerf,SNames{loopIndex},[]);
            end
            allPerf.(SNames{loopIndex}) = [allPerf.(SNames{loopIndex}); perf.(SNames{loopIndex})];
        end
        time_all = [time_all toc(t) + time_offset];
        iter_count = [iter_count idx + iter_offset];

        fprintf(1,'Iteration %6d, Validation %s is %.4f, ', idx+iter_offset,valid_criteria,perf.(valid_criteria));
        disp(['norm of L is ' num2str(norm(L,'fro'))])

        % Increment dip_ctr if validation performance is worse than previous
        % best.
        % If dip_ctr == dip_thresh, then stop to prevent overfitting
        % By default, dip_thresh is set to 50, which is absurdly high so early
        % stopping essentially never occurs. Modify this value to experiment.

        if perf.(valid_criteria) > (best_perf)
            best_perf = perf.(valid_criteria);
            bestL = L;
            dip_ctr = 0;
        elseif perf.(valid_criteria) > (best_perf - tol)
            dip_ctr = 0;
        else
            dip_ctr = dip_ctr + 1;
        end
        disp(['Dip counter is ' num2str(dip_ctr)])
        if dip_ctr == dip_thresh
            %             L = bestL;
            disp(['Stopped at ' num2str(idx+iter_offset) ' iterations'])
            converged = 1;
            break
        end

    end



    idx_i = indices(counter);
    [idx_j, diff_x] = sample_relevant_index(indiv_similarity, Ytrain, sim, dif, idx_i);
    if isnan(idx_j),continue,end

    if sparse_matrix == 1
        x_ij = sparse(Xtrain(:,idx_i)-Xtrain(:,idx_j));
    else
        x_ij = Xtrain(:,idx_i)-Xtrain(:,idx_j);
    end
    d_ij = distance (L,x_ij);


    if strcmp(mode,'auc')
        beta = 1;
    elseif strcmp(mode,'warp')
        beta = floor(length(diff_x)*rank_thresh);
        if beta<1,beta=1;end
    else
        error('invalid mode')
    end


    % find violator
    num_samples = 0;
    while num_samples < beta
        num_samples = num_samples + 1;
        idx_k = diff_x(floor(rand*length(diff_x)+1));
        x_ik = Xtrain(:,idx_i) - Xtrain(:,idx_k);
        if sparse_matrix == 1
            x_ik = sparse(x_ik);
        end
        d_ik = distance (L,x_ik);
        if d_ik < (d_ij + 1)
            break;
        end
    end
    ns_all(idx) = num_samples;

    if manifold
        % find low rank gradient if violator found
        if d_ik < (d_ij + 1)
            rank1_x = floor(length(diff_x)/num_samples);
            loss_value = loss_table(rank1_x+1)/ loss_table(length(diff_x)+1);
            p(:,(grad_rank+1):(grad_rank+2)) = -stepsize* 1/batchsize * (1-lam) * loss_value  * [x_ij, -x_ik];
            q(:,(grad_rank+1):(grad_rank+2)) = [x_ij, x_ik];
            grad_rank = grad_rank + 2;

        end

        % find low rank gradient for regularizer
        reg_index = reg_indices(reg_counter);
        [reg_p,reg_q] =  get_regularizer_gradient(reg_index, x_ij,x_ij,L,regularizer);
        p(:,(grad_rank+1)) = -stepsize* 1/batchsize * lam * reg_p;
        q(:,(grad_rank+1)) = reg_q;
        grad_rank = grad_rank + 1;

        if reg_counter == reg_maxindex, reg_indices = randperm(reg_maxindex); reg_counter = 0; end
        reg_counter = reg_counter + 1;
    else
        if d_ik < (d_ij + 1)
            gradl = L'*x_ij*x_ij'-L'*x_ik*x_ik';
            gradient = (1/batchsize * ((1-lam)*gradl + lam * L' * x_ij *x_ij'))';
            avg_grad = avg_grad + gradient;
        else
            gradient = (1/batchsize * lam * L' * x_ij *x_ij')';
            avg_grad = avg_grad + gradient;
        end
    end


    %Update gradients
    minibatch_count = minibatch_count + 1;
    if minibatch_count == batchsize
        %If manifold, use retraction
        if manifold
            p = p(:,1:grad_rank);
            q = q(:,1:grad_rank);
            if grad_rank > 0
                [L,pL] = gradient_update_L(L, pL, p, q);
            end
            minibatch_count = 0;
            grad_rank = 0;
            p = zeros(dim, batchsize*3);
            q = zeros(dim, batchsize*3);
        else
            L = L - stepsize*avg_grad;
            avg_grad = zeros(size(L));
            minibatch_count = 0;
        end
    end

    counter = counter + 1;
    if counter == n + 1
        counter = 1;
        indices = randperm(n);
    end

end


% fill out output variables
L = bestL;
if ~isempty(time_all)
    Diag.time = time_all(end)-time_offset;
end
Diag.time_all = time_all;
Diag.iter_count = iter_count;
Diag.allPerf = allPerf;
Diag.params = runtime_params;
Diag.ns_all = ns_all(1:idx);
end


function result = distance(L,x)
Lpx = L'*x;
result = (Lpx)'*(Lpx);
end

function result = grad_L(L,x_ij,x_ik)
result = L*x_ij*x_ij'-L*x_ik*x_ik';
end

function [L,pL] = gradient_update_L(L, pL, p, q)
%pL = (L'*L) \ L';
%h1 = pL*p;
%h2 = pL*q;

LtL = L'*L;
h1 = LtL \ (L'*p);
h2 = LtL \ (L'*q);

s = h1'*h2;
h1h = L*h1;
L = L + (-.5*h1h + p +(-0.5*p + 3/8*h1h)*s)*h2';

end


function result = set_default(var,field,def)
if ~isfield(var,field)
    result = def;
else
    if isempty(var.(field))
        result = def;
    else
        result = var.(field);
    end
end
end

function [idx_j, diff_x] = sample_relevant_index(indiv_similarity, Ytrain, sim, dif, idx_i)

if indiv_similarity
    sim_x = sim{idx_i,1};
    if length(sim_x) == 0
        % x is the only one of its class, so no positive points exist
        idx_j = nan;
    else
        rand_index = floor(rand*length(sim_x)+1);
        idx_j =  sim_x(rand_index);
    end
    diff_x = dif{idx_i};

else
    sim_x = sim{Ytrain(idx_i)};
    if isscalar(sim_x)
        % x is the only one of its class, so no positive points exist
        idx_j = nan;
    else
        while 1
            rand_index = floor(rand*length(sim_x)+1);
            idx_j = sim_x(rand_index);% + class_start(classid_reverse_lookup);
            if idx_j ~= idx_i
                break
            end
        end
    end
    diff_x = dif{Ytrain(idx_i)};

end

end

function [reg_p, reg_q] = get_regularizer_gradient(reg_index, x_ij, x_ik, L, regularizer)
switch regularizer
    case char('lmnn')
        reg_p =  x_ij;
        reg_q =  x_ij;
    case char('l2')
        reg_p = L(:,reg_index);
        reg_q = reg_p;
    otherwise
        error('Invalid regularizer')
end
end


