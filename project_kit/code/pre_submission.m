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
train_split = 0.9;

train_indices = rand(length(price_train),1)<train_split;
price_train;
%price_train_set = price_train(train_indices,:);
%price_validation_set = price_train(~train_indices,:);
price_train_set = price_train;

%% find prominent features
boosted_set = boost_features([word_train bigram_train; word_test bigram_test],1000);
size([word_train bigram_train; word_test bigram_test])
size(boosted_set)
fprintf('Freeing some memory...\n')
clear word_train
fprintf('deleted word_train...')
clear word_test
fprintf('deleted word_test...')
clear bigram_train
fprintf('deleted bigram_train...')
clear bigram_test
fprintf('deleted bigram_test...\n')

%% pick our data
fprintf('Performing SVD...\n')
%[U,S,V]=svds([word_train bigram_train],100);
[U,S,V]=svds(boosted_set,100);
fprintf('Freeing some memory...\n')
clear boosted_set
fprintf('deleted boosted_set...')

full_data = U*S*V';

% select features
fprintf('Performing features selection...\n')
selected_data = select_features(full_data,floor(0.0492*size(full_data,1)));

fprintf('Freeing some memory...\n')
clear full_data
fprintf('deleted full_data...')

fprintf('Assigning training set data...\n')
data_train_set = selected_data(1:length(price_train),:);
fprintf('Assigning validation set data...\n')
data_validation_set = selected_data(length(price_train)+1:end,:);

% free some memory
fprintf('Freeing some memory...\n')
clear selected_data
fprintf('deleted selected_data...')

%data_train_set = selected_data(train_indices,:);
%data_validation_set = selected_data(~train_indices,:);

%% linear regression by city

%city_train_set = city_train(train_indices,:);
%city_validation_set = city_train(~train_indices,:);
% assign variables
fprintf('assigning city train set...\n')
city_train_set = city_train;
fprintf('assigning city validation set...\n')
city_validation_set = city_test;

fprintf('freeing some ram...\n')
clear city_train % free up some ram
fprintf('deleted city_train...')
clear city_test
fprintf('deleted city_test...\n')

learners = cell(7,1);
num_weak_learners = 1;
data = data_train_set;

clear data_train_set
fprintf('beginning learning...\n')
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
%rms_error = sqrt(sum((price_validation_set - prediction).^2)/length(price_validation_set))
prices = prediction;
%% generate prediction text file
dlmwrite('submit.txt',prices,'precision','%d');