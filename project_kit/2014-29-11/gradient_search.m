function price = gradient_search(datum, probs, step_size, start_price)
price = start_price;
min_prob = Inf;
left_prob = Inf;
right_prob = Inf;
done = 0;

%size(datum)
%size(probs)
log_p_vec = ((probs(1,:)-(price-step_size)).^2./(2*probs(2,:).^2) + log((sqrt(2*pi)*probs(2,:)))).*datum;
log_p_vec(find(isnan(log_p_vec) == 1)) = 0;
left_prob = sum(log_p_vec);

log_p_vec = ((probs(1,:)-(price+step_size)).^2./(2*probs(2,:).^2) + log((sqrt(2*pi)*probs(2,:)))).*datum;
log_p_vec(find(isnan(log_p_vec) == 1)) = 0;
right_prob = sum(log_p_vec);

if left_prob < right_prob
    direction = -1;
else
    direction = 1;
end

while ~done
    log_p_vec = ((probs(1,:)-price).^2./(2*probs(2,:).^2) + log((sqrt(2*pi)*probs(2,:)))).*datum;
    log_p_vec(find(isnan(log_p_vec) == 1)) = 0;
    log_probs = sum(log_p_vec);
    if log_probs < min_prob
        min_prob = log_probs;
        price = price + direction*step_size;
    else
        done = 1;
    end
end

%{
%prices = linspace(0,30,30/step_size);
%log_probs = zeros(size(prices));
for i = 1:length(prices)
    log_p_vec = (probs(1,:)-prices(i)).^2./(2*probs(2,:).^2).*datum;
    log_p_vec(find(isnan(log_p_vec) == 1)) = 0;
    log_probs(i) = sum(log_p_vec);
end
log_probs(log_probs == Inf) = 0;
[[], idx] = min(log_probs);
price = prices(idx(1));
%}