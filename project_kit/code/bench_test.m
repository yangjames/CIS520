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
train = 16000;

price_train;
price_train_set = price_train(1:train,:);
price_validation_set = price_train(train+1:end,:);

full_data = [city_train word_train bigram_train];

% select features
selected_data = select_features(full_data,1000);
data_train_set = selected_data(1:train,:);
data_validation_set = selected_data(train+1:end,:);

%% naive bayes
%model = fitNaiveBayes(full_train_set, price_train_set,'dist','mn');
%prediction = model.predict(full_validation_set);

%% boosted linear regression
num_weak_learners = 500;
data = data_train_set;
w = generate_lr_weak_learners(data,price_train_set,500);
% boost
bins = linspace(min(price_train_set), max(price_train_set),1000);
boost = lr_boost(price_train_set,w,bins,200);
%prediction = [data_validation_set ones(size(data_validation_set,1),1)]*mean(w,2);

%% calculate error
rms_error = sqrt(sum((price_validation_set - prediction).^2)/length(price_validation_set))