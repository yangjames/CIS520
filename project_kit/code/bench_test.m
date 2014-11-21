clear all
%% load data

tic;
start_time = toc;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

%% split training data up
train_split = 0.8;

train_indices = rand(length(price_train),1)<train_split;
price_train;
price_train_set = price_train(train_indices,:);
price_validation_set = price_train(~train_indices,:);

%% pick our data
%full_data = [city_train word_train bigram_train];

%full_data = [word_train];
[U,S,V]=svds([word_train bigram_train],100);
full_data = U*S*V';

% select features
selected_data = select_features(full_data,1000);
data_train_set = selected_data(train_indices,:);
data_validation_set = selected_data(~train_indices,:);

%% naive bayes
%model = fitNaiveBayes(full_train_set, price_train_set,'dist','mn');
%prediction = model.predict(full_validation_set);

%% linear regression by city

city_train_set = city_train(train_indices,:);
city_validation_set = city_train(~train_indices,:);
learners = cell(7,1);
num_weak_learners = 1;
data = data_train_set;
for i = 1:7
    city_indices = find(city_train_set(:,i) == 1);
    learners{i} = generate_lr_weak_learners(data(city_indices,:),price_train_set(city_indices,:),num_weak_learners);
    validation_city_indices = find(city_validation_set(:,i) == 1);
    prediction(validation_city_indices) = [data_validation_set(validation_city_indices,:) ones(length(validation_city_indices),1)]*mean(learners{i},2);
end
prediction = prediction';

%% cross validated linear regression
%{
num_weak_learners = 1;
data = data_train_set;
w = generate_lr_weak_learners(data,price_train_set,num_weak_learners);
fprintf('Predicting...\n')
prediction = [data_validation_set ones(size(data_validation_set,1),1)]*mean(w,2);
%}
%% calculate error
rms_error = sqrt(sum((price_validation_set - prediction).^2)/length(price_validation_set))