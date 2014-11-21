function data = select_features(data, threshold)

total_occurrences = sum(data,1);
weak_features = find(total_occurrences < threshold);
for i = 1:length(weak_features)
    if (mod(i, floor(length(weak_features)/100)) == 0) || i == length(weak_features)
        fprintf('features selection progress: %3.f\n',i/length(weak_features)*100);
    end
    data(:,weak_features(i)) = zeros(size(data,1),1);
end