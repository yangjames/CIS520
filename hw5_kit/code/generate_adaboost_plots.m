%% Written / report part of Question 1 for Homework 6

%% Put your written answers here.
clear all
answers{1} = 'AdaBoost does not seem to overfit. Typically, we would see a concave up parabolic relationship between error and iteration for training error. Instead, we see testing error converging to training error. This implies that with infinite iterations, AdaBoost will achieve zero training error.';
answers{2} = 'Margin decreases as T increases. This is because we will eventually achieve zero training error on the dataset. Since testing error follows training error, we will see that with increasing T, we see fewer incorrect marks, meaning that margin will decrease. This makes sense because at each iteration, we apply a stronger weight to the weak classifier with smallest error, pulling incorrect classifications towards the correct one.';
answers{3} = 'AdaBoost would get the two left examples incorrect because on the portions of the images that are white, there are more blue pixels for the number 3 and more red pixels for the number 5. This would mean that for these instances, the 5 vote outweighs the 3 vote on the image of the 3, and the 3 vote outweighs the 5 vote on the image with the 5.';

save('problem_answers.mat', 'answers');

%% Follow the instructions below to generate plots.

data = load('../data/mnist_all.mat');

[X Y] = get_digit_dataset(data, {'3', '5'}, 'train');
[Xtest Ytest] = get_digit_dataset(data, {'3', '5'}, 'test');

% Create weak learner pool for training and test data.
Yw = make_pixel_learners(X);
Yw_test = make_pixel_learners(Xtest);

%% Train AdaBoost for 200 rounds, and evaluate on the test data.
T = 200;

% Run adaboost for 200 arounds
boost = adaboost_train(Y, Yw, T);

% Compute test error and margins
[test_err margins] = adaboost_test(boost, Yw_test, Ytest);

%% Plot 1 - Training err, Test err vs. T

figure('Name', 'Train vs. Test Err of Adaboost');
hold on;
plot(1:T, boost.train_err, '--r', 'LineWidth', 2);
plot(1:T, test_err, '-b', 'LineWidth', 2);
hold off;
legend({'Train Err', 'Test Err'});
xlabel('T');
ylabel('Error');
title('Train err vs. Test error of Adaboost');

% Save plot to disk
print -djpeg -r72 plot_1.jpg

%% Plot 2 - Margin vs T
clear h;

trange = [5 25 200];
figure('Name', 'Margin distribution vs. T');
hold on;
cols = {'b','r','g'};
for i = 1:3
    yhat = margins{trange(i)};
    h(i) = cdfplot(yhat);
    
    set(h(i), 'LineWidth', 2);
    set(h(i), 'Color', cols{i});
    %line([mean(yhat) mean(yhat)], [0 1], 'LineStyle', '--', 'Color', cols{i});
end
hold off;

% Make graph nicely labelled and legend'ed
xlabel('Margin on Test Data');
ylabel('P(Margin >= x)');
xlim([-1 1]);
title('CDF of Margin on Test Data');
legend(h, arrayfun(@(x)sprintf('T=%d', x), trange, 'UniformOutput', false));

% Save plot to disk
print -djpeg -r72 plot_2.jpg

%% Plot 3 - Understanding Mistakes
figure('Name', 'Worst vs. Best Test Example');

idx5 = find(Ytest==1);
idx1 = find(Ytest==-1);

% Look at all examples of 3's, and find the examples with lowest and
% highest margin.
[worst_margins, sort_order] = sort(margins{end}(idx1));

subplot(2,2,1);
i = 1;
plot_boost_digit(boost, Xtest(idx1(sort_order(i)),:), 50);
title(sprintf('margin = %.2f', worst_margins(i)));

subplot(2,2,2);
i = numel(idx1);
plot_boost_digit(boost, Xtest(idx1(sort_order(i)),:), 50);
title(sprintf('margin = %.2f', worst_margins(i)));

% Now do the same, but only looking at examples of 5's.
[worst_margins, sort_order] = sort(margins{end}(idx5));

subplot(2,2,3);
i = 1;
plot_boost_digit(boost, Xtest(idx5(sort_order(i)),:), 50);
title(sprintf('margin = %.2f', worst_margins(i)));

subplot(2,2,4);
i = numel(idx5);
plot_boost_digit(boost, Xtest(idx5(sort_order(i)),:), 50);
title(sprintf('margin = %.2f', worst_margins(i)));

print -djpeg -r72 plot_3.jpg

