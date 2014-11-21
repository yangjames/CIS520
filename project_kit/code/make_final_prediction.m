function prediction = make_final_prediction(model, city_validation_set)

% Input
% X_test : a 1xp vector representing "1" test sample.
% X_test=[city word bigram] a 1-by-10007 vector
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.
data_validation_set = model.transformed_test;
for i = 1:7
    validation_city_indices = find(city_validation_set(:,i) == 1);
    prediction(validation_city_indices) = [data_validation_set(validation_city_indices,:) ones(length(validation_city_indices),1)]*mean(model.learners{i},2);
end
prediction = prediction';
%prediction = model.predict(X_test);