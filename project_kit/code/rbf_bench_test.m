clear all
%% load data

tic;
start_time = toc;
load ../data/city_train.mat
%load ../data/city_test.mat
load ../data/word_train.mat
%load ../data/word_test.mat
load ../data/bigram_train.mat
%load ../data/bigram_test.mat
load ../data/price_train.mat

%% split training data up
train_split = 0.8; % percentage that will be training
train_indices = rand(length(price_train),1)<train_split;
price_train_set = price_train(train_indices,:);
price_validation_set = price_train(~train_indices,:);

%% pick our data
full_data = generate_rbf([word_train bigram_train],price_train);
clear word_train bigram_train

%% find prominent features
%{
fprintf('Performing SVD...\n')
[U,S,V]=svds(rbf_data,100);
clear rbf_data

fprintf('Generating full dataset...\n')
full_data = U*S*V';
clear U S V
%}
fprintf('Assigning training and validation data...\n')
data_train_set = full_data(train_indices,:);
data_validation_set = full_data(~train_indices,:);
clear full_data

%% linear regression by city
%{
% assign variables
fprintf('assigning city train set...\n')
city_train_set = city_train(train_indices,:);
fprintf('assigning city validation set...\n')
city_validation_set = city_train(~train_indices,:);

fprintf('Freeing some memory...\n')
clear city_train % free up some ram
fprintf('deleted city_train...')
%clear city_test
%fprintf('deleted city_test...\n')

learners = cell(7,1);
num_weak_learners = 1;
data = data_train_set;

clear data_train_set
fprintf('begin learning...\n')
for i = 1:7
    city_indices = find(city_train_set(:,i) == 1);
    learners{i} = generate_lr_weak_learners(data(city_indices,:),price_train_set(city_indices,:),num_weak_learners);
    validation_city_indices = find(city_validation_set(:,i) == 1);
    prediction(validation_city_indices) = [data_validation_set(validation_city_indices,:) ones(length(validation_city_indices),1)]*mean(learners{i},2);
end
current_stamp = toc;
end_time = (current_stamp-start_time);
fprintf('Finished training in %dm %3.3fs\n', floor(end_time/60), mod(end_time,60));

for i = 1:7
    validation_city_indices = find(city_validation_set(:,i) == 1);
    prediction(validation_city_indices) = [data_validation_set(validation_city_indices,:) ones(length(validation_city_indices),1)]*mean(learners{i},2);
end
prediction = prediction';
end_time = (toc-current_stamp);
fprintf('Finished predicting in %dm and %3.3fs\n',floor(end_time/60), mod(end_time,60));
%}

%% linear regression
num_weak_learners = 1;
data = data_train_set;

w = generate_lr_weak_learners(data,price_train_set,num_weak_learners);

current_stamp = toc;
end_time = (current_stamp-start_time);
fprintf('Finished training in %dm %3.3fs\n', floor(end_time/60), mod(end_time,60));
fprintf('Predicting...\n')
end_time = (toc-current_stamp);
prediction = [data_validation_set ones(size(data_validation_set,1),1)]*mean(w,2);
prediction(find(isnan(prediction))) = 0;
fprintf('Finished predicting in %dm and %3.3fs\n',floor(end_time/60), mod(end_time,60));

%% calculate error
rms_error = sqrt(sum((price_validation_set - prediction).^2)/length(price_validation_set));
fprintf('RMS error: %6.6f\n', rms_error);