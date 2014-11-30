clear all
%% load data

tic;
start_time = toc;
%load ../data/city_train.mat
load ../data/word_train.mat
load ../data/bigram_train.mat
load ../data/price_train.mat

good_indices = find(sum(word_train,2) > 50);
full_x = [word_train(good_indices,:) bigram_train(good_indices,:)];
clear word_train bigram_train
%{
fprintf('Performing SVD...\n')
[U,S,V] = svds(full_x,150);
fprintf('Generating transformed dataset...\n')
prin_comps = 90;
fprintf('Using %d principal copmonents...\n',prin_comps)
fprintf('PCA progress:   0%%')
z = zeros(size(full_x,1),prin_comps);
for i = 1:size(z,1)
    fprintf('\b\b\b\b%3.f%%',i/size(z,1)*100)
    z(i,:) = full_x(i,:)*V(:,1:prin_comps);
end
clear U S V full_x
%}
%% split training and validation data
train_split = 0.8;

train_indices = rand(length(price_train(good_indices)),1)<train_split;
price_train_set = price_train(train_indices,:);
price_validation_set = price_train(~train_indices,:);
clear price_train

train_x = full_x(train_indices,:);
val_x = full_x(~train_indices,:);

%% get conditional probability parameters
probs = generate_prob(train_x, price_train_set);

%% predict prices
prediction = zeros(size(price_validation_set));
tic;
prediction_start_time = toc;
fprintf('Prediction progress:   0%%, time elapsed:       0%%');
for i = 1:size(val_x,1)
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%3.f%%, time elapsed: %2dm %2ds',i/size(val_x,1)*100, floor(toc/60), mod(floor(toc-start_time),60));
    prediction(i) = gradient_search(full(val_x(i,:)),probs, 0.005, mean(price_train_set));
end
fprintf('\n');

rms_error = sqrt(sum((price_validation_set - prediction).^2)/length(price_validation_set));
fprintf('RMS error: %6.6f\n', rms_error);