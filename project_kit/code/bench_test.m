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
train_split = 0.8;

train_indices = rand(length(price_train),1)<train_split;
price_train;
price_train_set = price_train(train_indices,:);
price_validation_set = price_train(~train_indices,:);

%% pick our data
fprintf('Performing SVD...\n')
[U,S,V]=svds([word_train bigram_train],90);
clear word_train bigram_train

%% find prominent features
fprintf('Generating data from SVD...\n')
full_data = U*S*V';
clear U S V

% select features
fprintf('Performing features selection...\n')
selected_data = select_features(full_data,floor(0.0492*size(full_data,1)));

clear full_data
%selected_data = boost_features(selected_data,1500);

fprintf('Assigning training set data...\n')
data_train_set = selected_data(train_indices,:);

fprintf('Assigning validation set data...\n')
data_validation_set = selected_data(~train_indices,:);
clear selected_data

%% linear regression by city

% assign variables
fprintf('assigning city train set...\n')
city_train_set = city_train(train_indices,:);
fprintf('assigning city validation set...\n')
city_validation_set = city_train(~train_indices,:);
clear city_train

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

%% cross validated linear regression
%{
num_weak_learners = 1;
data = data_train_set;
w = generate_lr_weak_learners(data,price_train_set,num_weak_learners);
fprintf('Predicting...\n')
prediction = [data_validation_set ones(size(data_validation_set,1),1)]*mean(w,2);
%}
%% calculate error
rms_error = sqrt(sum((price_validation_set - prediction).^2)/length(price_validation_set));

end_time = (toc-current_stamp);
fprintf('Finished predicting in %dm and %3.3fs\n',floor(end_time/60), mod(end_time,60));
fprintf('RMS error: %6.6f\n', rms_error);
figure(1)
clf
plot(prediction, price_validation_set,'r*')
set(gca, 'xlim', [0 20], 'ylim',[0 20])
axis equal
xlabel('predicted price')
ylabel('actual price')
title('validation vs. prediction')