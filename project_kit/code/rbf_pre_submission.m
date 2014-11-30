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

%% get training data
price_train_set = price_train;

%% pick our data
[rbf, full_data] = generate_rbf([word_train bigram_train],price_train);
test_data = [word_test bigram_test];
clear word_train bigram_train word_test bigram_test city_train city_test

%% find prominent features
fprintf('Assigning training and validation data...\n')
data_train_set = full_data;
data_validation_set = zeros(size(test_data));
for i = 1:size(data_validation_set,2)
    data_validation_set(:,i) = exp(-(rbf(1,i) - price_train).^2/(2*rbf(2,i)^2));
end
clear full_data

%% cross validated linear regression
num_weak_learners = 1;
data = data_train_set;
w = generate_lr_weak_learners(data,price_train_set,num_weak_learners);
fprintf('Predicting...\n')
prediction = [data_validation_set ones(size(data_validation_set,1),1)]*mean(w,2);
nan_idx = isnan(prediction);
prediction(find(nan_idx == 1)) = 0;

prices = prediction;
%% generate prediction text file
dlmwrite('submit.txt',prices,'precision','%d');