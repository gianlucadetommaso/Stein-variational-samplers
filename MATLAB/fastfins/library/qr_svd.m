function [U, s] = qr_svd(X, tol)
%QR_SVD
%
%   This one is not fast!!

[Q,R]       = qr(X, 0);
[Phi,S,~]   = svd(R, 'econ');
[s,ind]     = sort(diag(S), 'descend');
jnd         = s>=tol; 
U           = Q*Phi(:,ind(jnd));
s           = s(jnd);

end