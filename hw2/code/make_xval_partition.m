function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

% even_bins=floor(n/n_folds);
% left=mod(n, n_folds);
% part=zeros(1,n);
% for i=1:even_bins
%     part((i-1)*n_folds+1:i*n_folds)=randperm(n_folds);
% end
% part(even_bins*n_folds+1:even_bins*n_folds+left)=floor(rand(1,left)*n_folds)+1;

part=mod(randperm(n),n_folds)+1;