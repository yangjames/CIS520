function [error] = knn_xval_error(K, X, Y, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(K, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KNN_TEST).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KNN_TEST

% FILL IN YOUR CODE HERE

if (nargin < 5)
    distFunc='l2';
end

num_bins=max(part);
epsilon=zeros(num_bins,1);
for bin=1:num_bins
    correct_labels=Y(part==bin);
    xval_data=X(part==bin,:);
    training_data=X(part~=bin,:);
    training_labels=Y(part~=bin);
    test_labels=sign(knn_test(K, training_data, training_labels, xval_data, distFunc));
    epsilon(bin)=mean(test_labels ~= correct_labels);
end
error = mean(epsilon);