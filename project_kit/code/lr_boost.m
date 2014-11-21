function boost = lr_boost(Y,Yw,bins,T)

err = zeros(1,T);
alpha = zeros(1,T);
h = zeros(1,T);
train_err = zeros(1,T);
y_hat = zeros(size(Y));

D=ones(size(Y))./size(Y,1);
size(Y)
size(Yw)
difference = abs(bsxfun(@minus, Y, Yw));

for t = 1:T
    errors = D'*difference;
    [err(t), h(t)]=min(errors);
    alpha(t) = 1/2*log((1-err(t))/err(t));
    D = D.*exp(-alpha(t)*Y.*Yw(:,h(t)))/sum(D.*exp(-alpha(t)*Y.*Yw(:,h(t))));
    y_hat = y_hat + alpha(t)*Yw(:,h(t));
    train_err(t) = sqrt(sum((y_hat -Y).^2)/length(Y));
end
    
% Store results of boosting algorithm.
boost.train_err = train_err;
boost.err = err;
boost.h = h;
boost.alpha = alpha;