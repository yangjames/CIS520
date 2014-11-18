%% Plots/submission for SVM portion, Question 1.

%% Put your written answers here.
clear all
answers{1} = ['The intersection kernel works best.' ...
        ' This works best because from a bag-of-words model,' ...
        ' words are more likely to occur in the context they are used.' ...
        ' The intersection kernel takes two instances and creates a feature' ...
        ' space with the least used words in each email. Common words' ...
        ' will have high density and will be easily classified, but' ...
        ' more unlikely words such as email addresses will be more' ...
        ' scrutinized and better categorized.'];

save('problem_1_answers.mat', 'answers');

%% Load and process the data.

load ../data/windows_vs_mac.mat;
[X Y] = make_sparse(traindata, vocab);
[Xtest Ytest] = make_sparse(testdata, vocab);

%% Bar Plot - comparing error rates of different kernels

% INSTRUCTIONS: Use the KERNEL_LIBSVM function to evaluate each of the
% kernels you mentioned. Then run the line below to save the results to a
% .mat file.

k_poly_linear = @(x,x2) kernel_poly(x, x2, 1);
k_poly_quadratic = @(x,x2) kernel_poly(x, x2, 2);
k_poly_cubic = @(x,x2) kernel_poly(x, x2, 3);
k_gaussian = @(x,x2) kernel_gaussian(x, x2, 20);
k_intersection = @(x,x2) kernel_intersection(x, x2);

results.linear = kernel_libsvm(X, Y, Xtest, Ytest, k_poly_linear);% ERROR RATE OF LINEAR KERNEL GOES HERE
results.quadratic = kernel_libsvm(X, Y, Xtest, Ytest, k_poly_quadratic);% ERROR RATE OF QUADRATIC KERNEL GOES HERE
results.cubic = kernel_libsvm(X, Y, Xtest, Ytest, k_poly_cubic);% ERROR RATE OF CUBIC KERNEL GOES HERE
results.gaussian = kernel_libsvm(X, Y, Xtest, Ytest, k_gaussian);% ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
results.intersect = kernel_libsvm(X, Y, Xtest, Ytest, k_intersection);% ERROR RATE OF INTERSECTION KERNEL GOES HERE

% Makes a bar chart showing the errors of the different algorithms.
algs = fieldnames(results);
for i = 1:numel(algs)
    y(i) = results.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Kernel');
ylabel('Test Error');
title('Kernel Comparisons');

print -djpeg -r72 plot_1.jpg;
