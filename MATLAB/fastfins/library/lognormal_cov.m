function [nu, X, V, d] = lognormal_cov(mu, A)
%LOGNORMAL
%
%   Tiangang Cui, 18/August


N       = size(A, 1);
Sigma   = A*A';
DS      = diag(Sigma);

tmp     = mu + 0.5*DS;
nu      = exp(tmp);



%A       = repmat(tmp, 1, N) + repmat(tmp', N, 1) + Sigma;

X       = exp(repmat(tmp, 1, N) + repmat(tmp', N, 1) + Sigma);

%T       = nu*nu';
%C       = T.*( exp(Sigma) - 1 );

[V, D]  = eig(X);
[d,ind] = sort(real(diag(D)), 'descend');

%figure; semilogy(d)

%jnd     = ( cumsum(d)/sum(d) ) <= (1 - 1E-10);
%d       = d(jnd);
V       = V(:,ind);


%plot(X)

end