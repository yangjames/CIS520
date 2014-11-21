function model = init_model(bigram_test,bigram_train,...
    city_test,city_train,...
    word_test,word_train,...
    price_train)
%% split training data up
price_train_set = price_train;
%price_validation_set = price_test;

%% pick our data
[U,S,V]=svds([word_train bigram_train; word_test bigram_test],100);
full_data = U*S*V';

% select features
selected_data = select_features(full_data,1000);
data_train_set = selected_data(1:length(price_train),:);
data_validation_set = selected_data(length(price_train)+1:end,:);

%% linear regression by city

city_train_set = city_train;
%city_validation_set = city_test;
learners = cell(7,1);
num_weak_learners = 1;
data = data_train_set;
for i = 1:7
    city_indices = find(city_train_set(:,i) == 1);
    learners{i} = generate_lr_weak_learners(data(city_indices,:),price_train_set(city_indices,:),num_weak_learners);
end
model.learners = learners;
model.transformed_test = data_validation_set;