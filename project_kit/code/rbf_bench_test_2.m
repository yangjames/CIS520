clear all
%% load data
tic;
start_time = toc;
load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat

%% split training data up
train_split = 0.8; % percentage that will be training
train_indices = rand(length(price_train),1)<train_split;
price_train_set = price_train(train_indices,:);
price_validation_set = price_train(~train_indices,:);

%% pick our data
full_data = [word_train bigram_train];
train_set = full_data(train_indices,:);
validation_set = full_data(~train_indices,:);
clear full_data

rbf_data = generate_rbf_3(full_data);
clear word_train bigram_train

fprintf('Assigning training and validation data...\n')
data_train_set = rbf_data(1:length(price_train_set),:);
data_validation_set = full_data(length(price_validation_set)+1:end,:);
clear full_data

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

fprintf('Finished predicting in %dm and %3.3fs\n',floor(end_time/60), mod(end_time,60));

%% calculate error
rms_error = sqrt(sum((price_validation_set - prediction).^2)/length(price_validation_set));
fprintf('RMS error: %6.6f\n', rms_error);