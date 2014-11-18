%% Script/instructions on how to submit plots/answers for question 2.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data: this loads X, Xnoisy, and Y.
load('../data/breast-cancer-data-fixed.mat');

%% 2.1
%{
error_trend = zeros(4,4);
for iteration=1:100
    indices=randperm(length(Y),400);
    Y_train=Y(indices);
    Y_test=Y(~ismember(1:length(Y),indices));
    X_train=X(indices,:);
    X_test=X(~ismember(1:size(X,1),indices),:);
    X_noisy_train=X_noisy(indices,:);
    X_noisy_test=X_noisy(~ismember(1:size(X,1),indices),:);

    data=[Y_train X_train];
    noisy_data=[Y_train X_noisy_train];
    K=1;
    N=[2 4 8 16];
    for n=1:length(N)
        part = make_xval_partition(length(Y_train), N(n));
        error = knn_xval_error(K, X_train, Y_train, part, 'l2');
        test_labels=knn_test(K, X_train, Y_train, X_test, 'l2');
        error_test = sum(test_labels~=Y_test)/length(Y_test);
        error_trend(1,n) = error_trend(1,n) + error;
        error_trend(2,n) = error_trend(2,n) + error_test;
        %%plot_data(i,
        
        part_noisy = make_xval_partition(length(Y_train), N(n));
        error_noisy = knn_xval_error(K, X_noisy_train, Y_train, part_noisy, 'l2');
        test_labels_noisy=knn_test(K, X_noisy_train, Y_train, X_noisy_test, 'l2');
        error_test_noisy = sum(test_labels_noisy~=Y_test)/length(Y_test);
        error_trend(3,n) = error_trend(3,n) + error_noisy;
        error_trend(4,n) = error_trend(4,n) + error_test_noisy;
    end
end
error_trend = error_trend/100
%}
N=[2 4 8 16];
K=1;
n_fold_error=zeros(100,length(N));
n_fold_error_noisy=zeros(100,length(N));
for i=1:100
    indices=randperm(length(Y),400);
    Y_train=Y(indices);
    Y_test=Y(~ismember(1:length(Y),indices));
    X_train=X(indices,:);
    X_test=X(~ismember(1:size(X,1),indices),:);
    X_noisy_train=X_noisy(indices,:);
    X_noisy_test=X_noisy(~ismember(1:size(X,1),indices),:);

    % generate n-fold error vectors
    for n=1:length(N)
        partition = make_xval_partition(length(Y_train), N(n));
        n_fold_error(i,n) = knn_xval_error(K, X_train, Y_train, partition, 'l2');
        n_fold_error_noisy(i,n) = knn_xval_error(K, X_noisy_train, Y_train, partition, 'l2');
    end
    
    % generate knn errors
end
y=mean(n_fold_error);
e = std(n_fold_error);
y_noisy=mean(n_fold_error_noisy);
e_noisy = std(n_fold_error_noisy);
figure(1);
errorbar(N,y,e);
title('Regular data');
xlabel('folds');
ylabel('error');
figure(2);
errorbar(N,y_noisy,e_noisy);
title('Noisy data');
xlabel('folds');
ylabel('error');
answers{1} = 'There is no trend in the data. The variance seems to decrease with increased number of folds, but otherwise, there is no significant difference between the calculated errors. The noisy data tends to have higher error than the regular data, but this is expected.';

% Plotting with error bars: first, arrange your data in a matrix as
% follows:
%
%  nfold_errs(i,j) = nfold error with n=j of i'th repeat
%  
% Then we want to plot the mean with error bars of standard deviation as
% folows: y = mean(nfold_errs), e = std(nfold_errs), x = [2 4 8 16].
% 
% >> errorbar(x, y, e);
%
% Along with nfold_errs, also plot errorbar for test error. This will 
% serve as measure of performance for different nfold-crossvalidation.
%
% To add labels to the graph, use xlabel('X axis label') and ylabel
% commands. To add a title, using the title('My title') command.
% See the class Matlab tutorial wiki for more plotting help.
% 
% Once your plot is ready, save your plot to a jpg by selecting the figure
% window and running the command:
%
% >> print -djpg plot_2.1-noisy.jpg % (for noisy version of data)
% >> print -djpg plot_2.1.jpg  % (for regular version of data)
%
% YOU MUST SAVE YOUR PLOTS TO THESE EXACT FILES.

%% 2.2
N=10;
K=[1 3 5 7 9 11 13 15];
sigma=[1 2 3 4 5 6 7 8];
n_fold_error=zeros(100,length(N));
n_fold_error_noisy=zeros(100,length(N));
for i=1:100
    indices=randperm(length(Y),400);
    Y_train=Y(indices);
    Y_test=Y(~ismember(1:length(Y),indices));
    X_train=X(indices,:);
    X_test=X(~ismember(1:size(X,1),indices),:);
    X_noisy_train=X_noisy(indices,:);
    X_noisy_test=X_noisy(~ismember(1:size(X,1),indices),:);

    % generate knn errors
    for k=1:length(K)
        partition = make_xval_partition(length(Y_train), N);
        n_fold_error(i,k) = knn_xval_error(K(k), X_train, Y_train, partition, 'l2');
        n_fold_error_noisy(i,k) = knn_xval_error(K(k), X_noisy_train, Y_train, partition, 'l2');
    end
    
    % generate kernel regression errors
    for s=1:length(sigma)
        partition = make_xval_partition(length(Y_train), N);
        kernreg_error(i,s) = kernreg_xval_error(sigma(s), X_train, Y_train, partition, 'l2');
        kernreg_error_noisy(i,s) = kernreg_xval_error(sigma(s), X_train, Y_train, partition, 'l2');
    end
end
% plot knn stuff
y=mean(n_fold_error);
e=std(n_fold_error);
y_noisy=mean(n_fold_error_noisy);
e_noisy = std(n_fold_error_noisy);
figure(3);
errorbar(K,y,e);
title('KNN: Regular data vs. K');
xlabel('K');
ylabel('error');
figure(4);
errorbar(K,y_noisy,e_noisy);
title('KNN: Noisy data vs. K');
xlabel('K');
ylabel('error');

% plot kernreg stuff
y=mean(kernreg_error);
e=std(kernreg_error);
y_noisy=mean(kernreg_error_noisy);
e_noisy=mean(kernreg_error_noisy);
figure(5);
errorbar(sigma,y,e);
title('Kernel Regression: Regular data vs. sigma');
xlabel('sigma');
ylabel('error');
figure(6);
errorbar(sigma, y_noisy, e_noisy);
title('Kernel Regression: Noisy data vs. sigma');
xlabel('sigma');
ylabel('error');

answers{2} = 'This is where your answer to 2.2 should go. Short and sweet is the key.';

% Save your plots as follows:
%
%  noisy data, k-nn error vs. K --> plot_2.2-k-noisy.jpg
%  noisy data, kernreg error vs. sigma --> plot_2.2-sigma-noisy.jpg
%  regular data, k-nn error vs. K --> plot_2.2-k.jpg
%  regular data, kernreg error vs. sigma --> plot_2.2-sigma.jpg

%% Finishing up - make sure to run this before you submit.
save('problem_2_answers.mat', 'answers');