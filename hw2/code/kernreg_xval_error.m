function [error] = kernreg_xval_error(sigma, X, Y, part, distFunc)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(SIGMA, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KERNREG_TEST).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNREG_TEST

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
    test_labels=sign(kernreg_test(sigma, training_data, training_labels, xval_data, distFunc));
    epsilon(bin)=mean(test_labels ~= correct_labels);
end
error = mean(epsilon);