function model = init_model()

model = [];

% Example:
% tmp = load('magic.mat');
% model.regW = tmp.w;
load('../data/price_train.mat');
load('../data/bigram_train.mat');
load('../data/word_train.mat');
load('../data/city_train.mat');

%16249

data = [word_train, bigram_train];

%feat select

data1 = full(data);
sum_data = sum(data1);
rem_feat = zeros(1, size(sum_data,2));
rem_feat(sum_data >= 1000) = 1;

model.remove_features = rem_feat;

data1 = data1(:, (sum_data >= 1000));
data1 = sparse(data1);
data = [data1];
train_set_x = data;
train_set_c = city_train;
train_set_y = price_train;
%PCA
%[x_coeff, x_score, x_latent] = pca(data);
%[B, idx] = sort(x_latent, 'descend');

[U,S,V] = svds(data, 100);
%rank = 1;
%data = U(1:size(data,1),1:rank)*S(1:rank,1:rank)*V(1:size(data,2),1:rank).';
data = U*S*V.';
%data = x_score(:, idx(1:i))*x_coeff(:, idx(1:i)).';

%randomize
n = size(data,1);
%scramble = [zeros(1, int64(80*n*1/100)), ones(1, int64(20*n*1/100))];
%scramble = reshape(scramble(randperm(n*1)), n, 1);

%train_set_x = reshape(data(randperm(n)), n, 1);

%{
train_set_x = data(scramble == 0,:);
train_set_y = price_train(scramble == 0,:);
train_set_c = city_train(scramble == 0,:);

test_set_x = data(scramble == 1,:);
test_set_y = price_train(scramble == 1,:);
test_set_c = city_train(scramble == 1,:);
%}
%{
train_set_x = full(train_set_x);
sum_dat = sum(train_set_x)
train_set_x = train_set_x(:, sum_dat >= 1000);
train_set_x = sparse(train_set_x);
%}
%data = [train_set_x];
%size(data)

y_pred = [];
y_act = [];

%train
disp 'training lin regs'


names = {'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'};

for i = 1:7
    city_x = train_set_x(full(train_set_c(:,i)) == 1, :);
    city_y =  train_set_y(full(train_set_c(:,i)) == 1, :);
    
    %city_x_test = test_set_x(full(test_set_c(:,i)) == 1, :);
    %city_y_test = test_set_y(full(test_set_c(:,i)) == 1, :);
    
    %knn = fitcknn(city_x, city_y);
    %y_pred = [y_pred;  predict(knn, city_x_test)];
    %disp 'training'
    linRegs = fitlm(city_x, city_y);
    model.names{i} = linRegs;
    %disp 'predict'
    %y_pred =[y_pred; predict(linRegs, city_x_test)];
   % y_act = [y_act; city_y_test];
    
    
end
%{
disp 'train';
knn = fitcknn(train_set_x, train_set_y);
disp 'test';
y_pred = predict(knn, test_set_x);
y_act = test_set_y;
%}
%{
disp 'train'
 linRegs = fitlm(train_set_x, train_set_y);
    disp 'predict'
    y_pred =[y_pred; predict(linRegs, test_set_x)];
    y_act = test_set_y;
%}
%{
diff = y_act - y_pred;
diff_sq = diff.^2;
sum_divide_N = sum(diff_sq) / length(y_act);
rmse = sqrt(sum_divide_N)
%}