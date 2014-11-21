function prediction = make_final_prediction(model,X_test)

% Input
% X_test : a 1xp vector representing "1" test sample.
% X_test=[city word bigram] a 1-by-10007 vector
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.

X_test = full(X_test);
city = X_test(1:7);
X_test = X_test(1, 8:10007);

X_test = X_test(:, (model.good_features ~= 0));
X_test = sparse(X_test);


i = find(city == 1);
mod = model.names{1,i};
prediction_vec = predict(mod, X_test);
prediction = prediction_vec(1)