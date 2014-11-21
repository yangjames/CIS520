function model = init_model()

%% load data
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

%% assign various training data with cross validation
train = 16000;

price_train;
price_train_set = price_train(1:train,:);
price_validation_set = price_train(train+1:end,:);

full_data = [city_train word_train bigram_train];
full_data
full_validation_set = full_data(1:train,:);
full_test_set = full_data(train+1:end,:);

city_train;
city_train_set = city_train(1:train, :);
city_validation_set = city_train(train+1:end,:);

word_train;
word_train_set = word_train(1:train, :);
word_validation_set = word_train(train+1:end,:);

bigram_train;
bigram_train_set = bigram_train(1:train,:);
bigram_validation_set = bigram_train(train+1:end,:);

%% pca and naive bayes
%[train_coeff train_score train_latent] = pca(full_data);
%fprintf('done with pca')

%% naive bayes
%model = fitNaiveBayes(full_train_set, price_train_set,'dist','mn');
%prediction = model.predict(full_validation_set);

%% linear regression
%{
num_weak_learners = 10000;
data = full_train_set;
w = zeros(size(full_data,2),num_weak_learners);
partitions = mod(randperm(size(data,1)),num_weak_learners)+1;
tic;
start_time = toc;
prev_time = toc;
for i = 1:num_weak_learners
    % obtain indices of our partitioned data
    weak_learner_data_indices = find(partitions == i);
    
    % partition the data
    partitioned_data = data(weak_learner_data_indices,:);
    
    % partition the labels
    partitioned_labels = price_train_set(weak_learner_data_indices);
    
    % generate weak learner
    w(:,i) = (partitioned_data'*partitioned_data)\(partitioned_data'*partitioned_labels);
    current_time_stamp = toc;
    fprintf('perecent done: %3.2f%%. rate: %3.0f iterations per second. elapsed time: %6.3f seconds\n',i/num_weak_learners*100, 1/(current_time_stamp-prev_time), current_time_stamp-start_time);
    prev_time = current_time_stamp;
end
prediction = mean(full_validation_set*w,2);
%}

%% calculate error
%rms_error = sqrt(sum((price_validation_set - prediction).^2)/length(price_validation_set))