q=0:0.01:1;
p=(1-q).^2./(q.^2+(1-q).^2);
figure(1)
clf
plot(q,p,'r.')
xlabel('q')
ylabel('p')
title('p vs. q')
hold on
plot(q,1-q,'b.')
plot([0.5 0.5], [0 1],'k-')
text(.715,.15,'Incorrect')
text(.20,.83,'Correct')
l=p-(1-q);
p2=1-q;
%H=area([p(1:floor(end/2));p2(1:floor(end/2))]);
