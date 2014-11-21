clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

X_train =[city_train word_train bigram_train];
Y_train = price_train;
X_test = [city_test word_test bigram_test];

X_test = full(X_test);
city = X_test(:,1:7);
X_test = X_test(:, 8:10007);

[U,S,V] = svds(X_test, 100);
X_test = U*S*V.';

X_test = [city, X_test];
X_test = sparse(X_test);
initialize_additional_features;

%% Run algorithm
% Example by lazy TAs
model = init_model();
prices = zeros(size(X_test, 1), 1);
for i = 1:size(X_test,1)
    if (mod(i, 1000) == 0)
        disp 'next thou';
    end
   prices(i) = make_final_prediction(model, X_test(i,:));
end


%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');