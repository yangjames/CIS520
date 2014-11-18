function [y] = nb_test(nb, X)
% Generate predictions for a Gaussian Naive Bayes model.
%
% Usage:
%
%   [Y] = NB_TEST(NB, X)
%
% X is a N x P matrix of N examples with P features each, and NB is a struct
% from the training routine NB_TRAIN. Generates predictions for each of the
% N examples and returns a 0-1 N x 1 vector Y.
% 
% SEE ALSO
%   NB_TRAIN

% YOUR CODE GOES HERE (compute log_p_x_and_y)
p_x_given_y0 = normpdf(X,repmat(nb.mu_x_given_y(:,1)',size(X,1),1),repmat(nb.sigma_x',size(X,1),1));
p_x_given_y1 = normpdf(X,repmat(nb.mu_x_given_y(:,2)',size(X,1),1),repmat(nb.sigma_x',size(X,1),1));

log_p_x_and_y=[log10(1-nb.p_y)+sum(log10(p_x_given_y0),2)';
                log10(nb.p_y)+sum(log10(p_x_given_y1),2)'];

% Take the maximum of the log generative probability 
[~, y] = max(log_p_x_and_y', [], 2);
% Convert from 1,2 based indexing to the 0,1 labels
y = y -1;