function data = boost_features(data, threshold)
original_size = size(data);
total_occurrences = sum(data,1);
strong_features = find(total_occurrences > threshold);
fprintf('Features boosting progress:   0%%');
for i = 1:length(strong_features)
    if (mod(i, floor(length(strong_features)/100)) == 0) || i == length(strong_features)
        fprintf('\b\b\b\b%3.f%%',i/length(strong_features)*100);
    end
    data(:,original_size(2)+i) = data(:,strong_features(i)).^2;
end
fprintf('\n');