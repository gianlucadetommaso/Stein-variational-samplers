function [nu, X, V, d] = logtrnormal_cov(mu, A, a)
%LOGTRNORMAL_COV
%
%   Tiangang Cui, 18/August


[N,r]       = size(A);

nor         = erf(a/sqrt(2))*2;
logG        = zeros(N);
for k = 1:r
    tmp     = repmat(A(:,k), 1, N) + repmat(A(:,k)', N, 1);
    logEk   = log( erf((a - tmp)/sqrt(2)) + erf((a + tmp)/sqrt(2)) ) - log(nor);
    logGk   = 0.5*tmp.^2 + logEk;
    logG    = logG + logGk;
end

X           = exp( logG + repmat(mu, 1, N) + repmat(mu', N, 1) );
nu          = exp( mu + sum( 0.5*A.^2 + log( erf((a - A)/sqrt(2)) + erf((a + A)/sqrt(2)) ) - log(nor), 2) );

[V, D]      = eig(X);
[d,ind]     = sort(real(diag(D)), 'descend');
V           = V(:,ind);

%figure; semilogy(d)

%jnd     = ( cumsum(d)/sum(d) ) <= (1 - 1E-10);
%d       = d(jnd);


%plot(X)

end