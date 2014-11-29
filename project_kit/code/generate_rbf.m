function rbf = generate_rbf(data, labels)
rbf = zeros(size(data));

fprintf('RBF generation progress:   0%%');
for i = 1:size(rbf,2)
    fprintf('\b\b\b\b%3.f%%',i/size(rbf,2)*100);
    feature = find(data(:,i) ~= 0);
    if (~isempty(feature))
        center_feature = mean(labels(feature));
        variance_feature = std(labels(feature));
        rbf(feature,i) = exp(-norm(center_feature-data(feature,i),2)/(2*variance_feature^2));
    end
end
fprintf('\n');