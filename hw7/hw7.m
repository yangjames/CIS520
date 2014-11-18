clear all
%% load data
load('data/ocr_data.mat')

[coeff, score, latent] = pca(testset.pixels);

%% #1
x_indices = find(testset.letter == 24);
y_indices = find(testset.letter == 25);
w_indices = find(testset.letter == 23);

x_data = score(x_indices,1:2);
y_data = score(y_indices,1:2);
w_data = score(w_indices,1:2);

figure(1)
clf
hold on
grid on
axis equal
title('x and y')
ylabel('PC2')
xlabel('-PC1')
plot(x_data(:,1),x_data(:,2),'b+');
plot(y_data(:,1),y_data(:,2),'go');

figure(2)
clf
hold on
grid on
axis equal
title('x and w')
ylabel('PC2')
xlabel('-PC1')
plot(-x_data(:,1),x_data(:,2),'go');
plot(-w_data(:,1),w_data(:,2),'b+');
drawnow
%% #2
figure(3)
clf
hold on
grid on
xlabel('number of PCs')
ylabel('Accuracy')
title('Accuracy vs number of PCs')
ratio = [];
data = [testset.pixels; trainset.pixels];
[coeff_full, score_full, latent_full] = pca(data);
for i = 1:size(data,2)
    comps = i;
    prin_comps = [score_full(:,1:comps) zeros(size(score_full,1),size(score_full,2)-comps)];
    reproj = prin_comps*coeff_full';
    difference = reproj-bsxfun(@minus,data, mean(data,1));
    err_fro = norm(difference,'fro');
    err_fro_orig = norm(bsxfun(@minus,data, mean(data,1)),'fro');
    ratio(i) = err_fro^2/err_fro_orig^2;
end
plot(1:64,1-ratio,'k*-')
drawnow
%% #3
figure(4)
dimensions = [5 10 20];
train_data = [trainset.pixels; testset.pixels];
train_labels = [trainset.letter; testset.letter];
[train_coeff,train_score,train_latent]=pca(train_data);

test_score = train_score(size(trainset.pixels,1)+1:end,:);
train_score = train_score(1:size(trainset.pixels,1),:);
for i = 1:length(dimensions)
    comps = dimensions(i);
    prin_score = train_score(:,1:comps);
    model = fitNaiveBayes(prin_score,trainset.letter);
    predictions = model.predict(test_score(:,1:comps));
    
    error = sum(predictions ~= testset.letter);
    accuracies(i)=1-sum(predictions ~= testset.letter)/length(testset.letter);
    
end
bar(accuracies)

xlabel('number of principal components')
ylabel('accuracy')
title('Naive Bayes on PCA')

%% #4
clear all
% load data
load('data/ocr_data.mat')

figure(5)
dimensions = [5 10 20];
train_data = [trainset.pixels; testset.pixels];
train_labels = [trainset.letter; testset.letter];
[train_coeff,train_score,train_latent]=pca(train_data);

for i = 1:length(dimensions)
    labels=[];
    error = [];
    distances = [];
    comps = dimensions(i);
    prin_score = train_score(:,1:comps);
    [indices,centroids]=kmeans(prin_score,26,'MaxIter',500);
    for j = 1:26
        clustered_points = find(indices == j);
        label(j) = mode(train_labels(clustered_points));
    end
    for j = size(trainset.pixels,1)+1:size(train_data,1)
        prediction(j-size(trainset.pixels,1)) = label(indices(j));
    end
    
    error = (prediction' ~= testset.letter);
    accuracy(i)=1-sum(error)/length(testset.letter);
end
bar(accuracy)

xlabel('number of principal components')
ylabel('accuracy')
title('K-means on PCA')