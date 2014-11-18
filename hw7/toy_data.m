%% k mean
% data points
data = [0 0;...
    1 0;...
    0 1;...
    2 3;...
    3 2;...
    3 3;...
    22 3];

% number of clusters
K=2;

% initialized centroids
mu = [3 3; 3 2];
prev_mu = [0 0; 0 0];

% convergence flag
converged = 0;
r=[];
while ~converged
    % assign points to nearest centroid
    for i = 1:size(data,1)
        d = sum(bsxfun(@minus,mu,data(i,:)).^2,2);
        [value index] = min(d);
        r(i,1) = 1==index;
        r(i,2) = 2==index;
    end
    
    %update
    mu = (data'*r)./repmat(sum(r,1),2,1);
    
    % check if mu_1 is the same as before
    if sum(mu(1,:) == prev_mu(1,:))/2 == 1
        % check if mu_2 is the same as before
        if sum(mu(2,:) == prev_mu(2,:))/2 == 1
            converged = 1;
        else
            prev_mu(1,:) = mu(1,:);
            prev_mu(2,:) = mu(2,:);
        end
    else
        prev_mu(1,:) = mu(1,:);
        prev_mu(2,:) = mu(2,:);
    end
end

%% k medoids
% data points
data = [0 0;...
    1 0;...
    0 1;...
    2 3;...
    3 2;...
    3 3;...
    22 3];

iterator = 1;
for i = 1:size(data,1)-1
    for j= i+1:size(data,1)
        d(iterator) = norm(data(i,:)-data(j,:));
        iterator = iterator+1;
    end
end
