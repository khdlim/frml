function [sim,diff] = get_sim_diff(indiv_similarity,target_k, Xtrain,Ytrain)
%forms the sets sim and diff for use with lrml_warp
%
% indiv_similarity = 1 :
% Every training point has its own indices. Use this if Ytrain is given in
% cell format for ranking problems. Also use this in classification problems
% if you want to use only the target_k nearest same-class neighbours by Euclidean
% distance as targets. This generates Nx1 sim/diff cell arrays.
%
%indiv_similarity = 0 :
% Only used for classification problems. In this case, target_k is ignored.
% For each query, similar points are considered to be all same-class points and
% different points are considered to be all different-class points.
% sim and diff are generated appropriately for this scenario for use with
% lrml_warp.m, with Ytrain in Option 1 format (see lrml_warp.m for details).
% This generates Kx1 sim and diff cell arrays.

if indiv_similarity
    n = size(Xtrain,2);
    sim = cell(n,1);
    diff = cell(n,1);
    %     for i = 1:n;
    if iscell(Ytrain)
        if size(Ytrain,2) == 1
            for i = 1:n
                sim{i} = Ytrain{i};
                diff{i} = setdiff((1:n)',[sim{i};i]);
            end
        else
            for i = 1:n
                sim{i} = Ytrain{i,1};
                diff{i} = Ytrain{i,2};
                %             diff{i} = setdiff((1:n)',[sim{i};i]);
            end
        end
    else
        %Set closest target_k neighbours by euclidean distance to be similar
        [D, I]  = get_distance_matrix(Xtrain, Xtrain);
        Labels  = Ytrain(I);
        Agree   = bsxfun(@eq, Ytrain', Labels);
        for i = 1:n;
            diff{i} = find(Ytrain ~= Ytrain(i));
            I_i = I(:,i);
            sim{i} = setdiff(I_i(find(Agree(:,i),target_k+1)),i);
            sim{i} = setdiff(sim{i},i);
        end

    end


else
    %classwise

    num_classes = max(Ytrain);

    sim = cell(num_classes,1);
    diff = cell(num_classes,1);

    %choose all same-class neighbors to be similar
    for i = 1:num_classes
        sim{i} = find(Ytrain == i);
        diff{i} = find(Ytrain ~= i);

    end

end
end


function [D,I] = get_distance_matrix(Xtrain, Xtest)

    [d, nTrain] = size(Xtrain);
    nTest = size(Xtest, 2);
	D = (bsxfun(@plus, dot(Xtest,Xtest)', dot(Xtrain,Xtrain)) - 2*Xtest'*Xtrain)';
    [v,I]   = sort(D, 1);
end
