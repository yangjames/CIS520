function [boost] = adaboost_train(Y, Yw, T)
% Runs AdaBoost for T rounds given the predictions of a set of weak learners.
%
% Usage:
%
%   boost = adaboost_train(Y, YW, T)
%
% Returns a struct containing the results of running AdaBoost for T rounds.
% The input Y should be a (-1,1) binary N x 1 vector of labels. The input
% YW should be a (-1,1) binary N x M matrix containing the predictions of M
% weak learners on the dataset. T is the number of rounds of boosting. The
% returned struct has the following fields:
%
%   boost.err - 1 x T vector of weighted error at round t
%   boost.train_err - 1 x T vector of cumulative training error
%   boost.h - 1 X T vector indicating which weak learner was chosen
%   boost.alpha - 1 X T vector of weights for combining learners
%
% NOTE: This implementation of Adaboost you are creating only works when
% you can precompute a pool of possible weak learners. In general, you
% might want to train a weak learner for each D rather than picking a
% precomputed one with minimal error.

% HINT: READ ALL THE HINTS AND DOCUMENTS WE GIVE YOU BEFORE BEGINNING.

% HINT: Look at ADABOOST_TEST before trying to implement AdaBoost so you
% can see how we expect this to be used.

% HINT: Precompute where each weak learner makes mistakes BEFORE running
% the main boosting loop. Then you can compute weighted error by weighting
% the mistakes of each learner.

% HINT: If predictions and labels are +1, -1, then errors are when the true 
% label multiplied by prediction is -1. 

% HINT: Follow the AdaBoost algorithm given in class, NOT THE ADABOOST
% ALGORITHM GIVEN IN BISHOP. DO NOT USE BISHOP FOR THIS.

% Perform any initialization or precomputation here.

% Initialize distribution over examples.

err = zeros(1,T);
alpha = zeros(1,T);
h = zeros(1,T);
train_err = zeros(1,T);
y_hat = zeros(size(Y));

D=ones(size(Y))./size(Y,1);
indicator = bsxfun(@ne, Y, Yw);

t0 = CTimeleft(T);
for t = 1:T
    t0.timeleft();
    
    % Compute the BEST weak learner according to current D, etc. -- put
    % AdaBoost logic here.  Make sure to update h(t), err(t), alpha(t), and
    % train_err(t). Note that h(t) should be the INDEX of the best weak
    % learner, and thus a single scalar number.

    % HINT: Make sure to normalize D so it sums to one!!
    
    errors = D'*indicator;
    [err(t), h(t)]=min(errors);
    alpha(t) = 1/2*log((1-err(t))/err(t));
    D = D.*exp(-alpha(t)*Y.*Yw(:,h(t)))/sum(D.*exp(-alpha(t)*Y.*Yw(:,h(t))));
    y_hat = y_hat + alpha(t)*Yw(:,h(t));
    train_err(t) = sum(sign(y_hat) ~= Y)/length(Y);
end
    
% Store results of boosting algorithm.
boost.train_err = train_err;
boost.err = err;
boost.h = h;
boost.alpha = alpha;