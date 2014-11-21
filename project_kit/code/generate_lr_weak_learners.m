function w = generate_lr_weak_learners(data,labels,num_weak_learners)
%num_weak_learners = 500;
%data = data_train_set;
w = zeros(size(data,2)+1,num_weak_learners);
partitions = mod(randperm(size(data,1)),num_weak_learners)+1;
for i = 1:num_weak_learners
    % obtain indices of our partitioned data
    weak_learner_data_indices = find(partitions == i);
    
    % partition the data
    partitioned_data = [data(weak_learner_data_indices,:) ones(length(weak_learner_data_indices),1)];
    
    % partition the labels
    partitioned_labels = labels(weak_learner_data_indices);

    % generate weak learner
    w(:,i) = (partitioned_data'*partitioned_data + speye(size(partitioned_data,2))*0.1)\(partitioned_data'*partitioned_labels);

    % print progress
    if mod(i,floor(num_weak_learners/100)) == 0 || i == num_weak_learners
        current_time_stamp = toc;
        fprintf('percent done: %3.f%%\n',i/num_weak_learners*100);
    end
end
