function rbf = generate_rbf_3(data)
mean_data = mean(data,1);
cov = std(data,0,1);
rbf = exp(-bsxfun(@minus,mean_data,data)./(2*cov.^2));