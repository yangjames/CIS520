function prob = generate_prob(data, labels)
prob = zeros(2,size(data,2));

fprintf('RBF generation progress:   0%%');
for i = 1:size(prob,2)
    fprintf('\b\b\b\b%3.f%%',i/size(prob,2)*100);
    feature = find(data(:,i) ~= 0);
    feature = find(labels(feature) ~= 0);
    if (~isempty(feature))
        prob(1,i) = mean(labels(feature));
        prob(2,i) = std(labels(feature));
    end
end
fprintf('\n');