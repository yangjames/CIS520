
%% Load the dataset.
load ../data/breast-cancer-data.mat

%% Part I - Learning Rates
clear answers

% Your plot of objective vs. iteration for learning rates should use the
% following step sizes:
step_size_range = [1 0.1 0.001 0.0001 1e-5]

% YOUR CODE GOES HERE
C=10^-3;
figure(1)
clf
colors = {'r','b','g','m','k'}
%[w obj grandnorm] = lr_train(X,Y,C,'step_size',1,'max_iter',5000);
weights=[];
gradnorms=[];
for step = 1:length(step_size_range)
    [w obj gradnorm]=lr_train(X,Y,C,'step_size',step_size_range(step),'max_iter', 5000, 'stop_tol',0);
    size(obj)
    plot(1:length(obj),obj,colors{step})
    hold on
    weights(step,:)=w;
    gradnorms(step)=gradnorm;
end

xlabel('Iteration')
ylabel('log10(obj)')
legend('step size: 1','step size: 0.1', 'step size: 0.001', 'step size: 0.0001', 'step size: 1e-5');
% Set a logarithmic scale using the following
set(gca,'YScale','log')

% Save with:
% print -djpeg -r72 step_sizes.jpg

answers{1} = 'Large step size causes massive fluctuations in objectives. By decreasing the step size, objectives converge to smaller and more stable values. However, at some point, the step size becomes too small and objectives start to increase again.';

Y_lr_test = [];
lr_errors=[];
for iteration=1:size(weights,1)
    Y_lr_test(:,iteration)=lr_test(weights(iteration,:),X);
    lr_errors(iteration)=sum(Y_lr_test(:,iteration) ~= Y);
end
figure(4)
clf
plot(lr_errors,'r-*')
xlabel('step size index')
ylabel('error')
title('error vs step size')
figure(5)
clf
plot(gradnorms,'b-*')
xlabel('step size index')
ylabel('gradnorm')
title('gradnorm vs step size')
drawnow
answers{2} = 'Step size of 0.001 had the smallest gradient, and step size 0.1 had the smallest training error. In general, optimizing the objective seems to correlate with optimizing training error.';

save('answers_1.mat', 'answers');

%% Part II - Learning Curves
clear answers

% For this section, generate the learning curves. Make sure to plot
% errorbars.

errors_nb = zeros(100,8);
errors_lr = zeros(100,8);
% YOUR CODE GOES HERE
for iteration = 1:100
    fprintf('Iteration: %d\n',iteration);
    % randomly separate the dataset into 80% training, 20% test
    indices = randperm(length(Y),floor(length(Y)*.8));

    Y_train = Y(indices);
    Y_test = Y(~ismember(1:length(Y),indices));
    X_train = X(indices,:);
    X_test = X(~ismember(1:size(X,1),indices),:);

    % Further subdivide the training set into 8 partitions
    partition = mod(randperm(size(X_train,1)),8)+1;

    % train NB and LR using partition 1 as training and record the test error
    partition_1 = find(partition == 1);

    nb1 = nb_train(X_train(partition_1,:),Y_train(partition_1));
    Y_nb_1 = nb_test(nb1, X_test);
    error_nb_1 = sum(Y_nb_1 ~= Y_test);

    [w_1 obj_1 gradnorm_1] = lr_train(X_train(partition_1,:),Y_train(partition_1),10^-3, 'step_size', 10^-3, 'stop_tol', 10^-5, 'max_iter', 1000);
    Y_lr_1 = lr_test(w_1, X_test);
    error_lr_1 = sum(Y_lr_1 ~= Y_test);

    % train NB and LR using partitions 1 and 2 as training and record the test
    % error.
    partition_12 = [find(partition == 1) find(partition == 2)];

    nb12 = nb_train(X_train(partition_12,:),Y_train(partition_12));
    Y_nb_12 = nb_test(nb12, X_test);
    error_nb_12 = sum(Y_nb_12 ~= Y_test);

    [w_12 obj_12 gradnorm_12] = lr_train(X_train(partition_12,:),Y_train(partition_12),10^-3,'step_size',10^-3,'stop_tol', 10^-5, 'max_iter', 1000);
    Y_lr_12 = lr_test(w_12, X_test);
    error_lr_12 = sum(Y_lr_12 ~= Y_test);

    % train NB and LR using partitions 1,2,and 3 as training and record the
    % test error.
    partition_123 = [find(partition == 1) find(partition == 2) find(partition == 3)];

    nb123 = nb_train(X_train(partition_123,:),Y_train(partition_123));
    Y_nb_123 = nb_test(nb123, X_test);
    error_nb_123 = sum(Y_nb_123 ~= Y_test);

    [w_123 obj_123 gradnorm_123] = lr_train(X_train(partition_123,:),Y_train(partition_123),10^-3,'step_size',10^-3,'stop_tol', 10^-5, 'max_iter', 1000);
    Y_lr_123 = lr_test(w_123, X_test);
    error_lr_123 = sum(Y_lr_123 ~= Y_test);

    % repeat for partitions (1,2,3,4), (1,2,3,4,5) and so forth.
    % 1 - 4
    partition_1234 = [find(partition == 1) find(partition == 2) find(partition == 3) find(partition == 4)];

    nb1234 = nb_train(X_train(partition_1234,:),Y_train(partition_1234));
    Y_nb_1234 = nb_test(nb1234, X_test);
    error_nb_1234 = sum(Y_nb_1234 ~= Y_test);

    [w_1234 obj_1234 gradnorm_1234] = lr_train(X_train(partition_1234,:),Y_train(partition_1234),10^-3,'step_size',10^-3,'stop_tol', 10^-5, 'max_iter', 1000);
    Y_lr_1234 = lr_test(w_1234, X_test);
    error_lr_1234 = sum(Y_lr_1234 ~= Y_test);

    % 1 - 5
    partition_12345 = [find(partition == 1) find(partition == 2) find(partition == 3) find(partition == 4) find(partition == 5)];

    nb12345 = nb_train(X_train(partition_12345,:),Y_train(partition_12345));
    Y_nb_12345 = nb_test(nb12345, X_test);
    error_nb_12345 = sum(Y_nb_12345 ~= Y_test);

    [w_12345 obj_12345 gradnorm_12345] = lr_train(X_train(partition_12345,:),Y_train(partition_12345),10^-3,'step_size',10^-3,'stop_tol', 10^-5, 'max_iter', 1000);
    Y_lr_12345 = lr_test(w_12345, X_test);
    error_lr_12345 = sum(Y_lr_12345 ~= Y_test);

    % 1 - 6
    partition_123456 = [find(partition == 1) find(partition == 2) find(partition == 3) find(partition == 4) find(partition == 5) find(partition == 6)];

    nb123456 = nb_train(X_train(partition_123456,:),Y_train(partition_123456));
    Y_nb_123456 = nb_test(nb123456, X_test);
    error_nb_123456 = sum(Y_nb_123456 ~= Y_test);

    [w_123456 obj_123456 gradnorm_123456] = lr_train(X_train(partition_123456,:),Y_train(partition_123456),10^-3,'step_size',10^-3,'stop_tol', 10^-5, 'max_iter', 1000);
    Y_lr_123456 = lr_test(w_123456, X_test);
    error_lr_123456 = sum(Y_lr_123456 ~= Y_test);

    % 1 - 7
    partition_1234567 = [find(partition == 1) find(partition == 2) find(partition == 3) find(partition == 4) find(partition == 5) find(partition == 6) find(partition == 7)];

    nb1234567 = nb_train(X_train(partition_1234567,:),Y_train(partition_1234567));
    Y_nb_1234567 = nb_test(nb1234567, X_test);
    error_nb_1234567 = sum(Y_nb_1234567 ~= Y_test);

    [w_1234567 obj_1234567 gradnorm_1234567] = lr_train(X_train(partition_1234567,:),Y_train(partition_1234567),10^-3,'step_size',10^-3,'stop_tol', 10^-5, 'max_iter', 1000);
    Y_lr_1234567 = lr_test(w_1234567, X_test);
    error_lr_1234567 = sum(Y_lr_1234567 ~= Y_test);

    % 1 - 8
    partition_12345678 = [find(partition == 1) find(partition == 2) find(partition == 3) find(partition == 4) find(partition == 5) find(partition == 6) find(partition == 7) find(partition == 8)];

    nb12345678 = nb_train(X_train(partition_12345678,:),Y_train(partition_12345678));
    Y_nb_12345678 = nb_test(nb12345678, X_test);
    error_nb_12345678 = sum(Y_nb_12345678 ~= Y_test);

    [w_12345678 obj_12345678 gradnorm_12345678] = lr_train(X_train(partition_12345678,:),Y_train(partition_12345678),10^-3,'step_size',10^-3,'stop_tol', 10^-5, 'max_iter', 1000);
    Y_lr_12345678 = lr_test(w_12345678, X_test);
    error_lr_12345678 = sum(Y_lr_12345678 ~= Y_test);
    
    errors_nb(iteration,:)=[error_nb_1 error_nb_12 error_nb_123 error_nb_1234 error_nb_12345 error_nb_123456 error_nb_1234567 error_nb_12345678];
    errors_lr(iteration,:)=[error_lr_1 error_lr_12 error_lr_123 error_lr_1234 error_lr_12345 error_lr_123456 error_lr_1234567 error_lr_12345678];
end

avg_errors_nb = mean(errors_nb,1);
avg_errors_lr = mean(errors_lr,1);

figure(2)
clf
plot(1:length(avg_errors_nb),avg_errors_nb,'r-*')
title('Learning Curve: Naive Bayes vs. Logistic Regression')
ylabel('Total error')
xlabel('Number of partitions')
hold on
plot(1:length(avg_errors_lr),avg_errors_lr,'b-*')
legend('Naive Bayes','Logistic Regression')
% Save with:
% print -djpeg -r72 learning_curves.jpg

answers{1} = 'Naive Bayes converges to peak test set performance faster. With the most data, logistic regression works better.'; 

save('answers_2.mat', 'answers');