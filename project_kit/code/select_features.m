function data = select_features(data, threshold)

total_occurrences = sum(data,1);
weak_features = find(total_occurrences < threshold);
fprintf('Features selection progress:   0%%');
for i = 1:length(weak_features)
    if (mod(i, floor(length(weak_features)/100)) == 0) || i == length(weak_features)
        fprintf('\b\b\b\b%3.f%%',i/length(weak_features)*100);
    end
    data(:,weak_features(i)) = zeros(size(data,1),1);
end
fprintf('\n');