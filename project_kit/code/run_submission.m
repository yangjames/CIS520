clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat

initialize_additional_features;

%% Run algorithm
model = init_model(bigram_test,bigram_train,...
    city_test,city_train,...
    word_test,word_train,...
    price_train);
prices = make_final_prediction(model,city_test);

%% Save results to a text file for submission
dlmwrite('submit.txt',prices,'precision','%d');