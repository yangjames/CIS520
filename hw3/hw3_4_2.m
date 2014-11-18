clear all
load('hw3-data/data.mat')

lambda=1;

L1_error=@(w) (norm(Y-X*w,2).^2)+lambda*norm(w,1);
L2_error=@(w) (norm(Y-X*w,2).^2)+lambda*(norm(w,2).^2);


L0_error_000=norm(Y).^2;
L0_error_001=@(w1) (norm(Y-X*[w1 0 0]').^2)+1;%(w1 ~= 0);
L0_error_010=@(w2) (norm(Y-X*[0 w2 0]').^2)+1;%(w2 ~= 0);
L0_error_011=@(w) (norm(Y-X*[w(1) w(2) 0]').^2)+2;%(w(1) ~= 0)+(w(2) ~= 0);
L0_error_100=@(w3) (norm(Y-X*[0 0 w3]').^2)+1;%(w3 ~= 0);
L0_error_101=@(w) (norm(Y-X*[w(1) 0 w(2)]').^2)+2;%(w(1) ~= 0) + (w(2) ~= 0);
L0_error_110=@(w) (norm(Y-X*[0 w(1) w(2)]').^2)+2;%(w(1) ~= 0) + (w(2) ~= 0);
L0_error_111=@(w) (norm(Y-X*w,2).^2)+3;%lambda*(sum(w(w ~= 0).^0));

clc
MLE=(X'*X)\X'*Y
[w_0_001, minerr_001]=fminsearch(L0_error_001,rand);
[w_0_010, minerr_010]=fminsearch(L0_error_010,rand);
[w_0_011, minerr_011]=fminsearch(L0_error_011,rand(2,1));
[w_0_100, minerr_100]=fminsearch(L0_error_100,rand)
[w_0_101, minerr_101]=fminsearch(L0_error_101,rand(2,1));
[w_0_110, minerr_110]=fminsearch(L0_error_110,rand(2,1));
[w_0_111, minerr_111]=fminsearch(L0_error_111,rand(3,1));
[minval,idx]=min([minerr_001, minerr_010, minerr_011, minerr_100, minerr_101, minerr_110, minerr_111]);
weights={w_0_001, w_0_010, w_0_011, w_0_100, w_0_101, w_0_110, w_0_111};

w_0=weights{idx}
w_1=fminsearch(L1_error,rand(3,1))
w_2=fminsearch(L2_error,rand(3,1))