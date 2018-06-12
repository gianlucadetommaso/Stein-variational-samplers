function J = explicit_J(model, HI)
%EXPLICIT_J
%
% Explicit Jacobian 
%
% Tiangang Cui, 20/Nov/2013

M       = (model.N/model.k);
N       =  model.N+1;
%off  = -1-dfdu(1:(end-1))*model.dt;
%udu  = sparse([1:N 2:N],[1:N 1:(N-1)], [ones(1,N), off(:)'], N, N);

RHS     = sparse((model.k+1):model.k:N, 1:M, ones(1,M), N, M);
tmp     = full(HI.udu'\RHS)';
J       = sqrt(model.dt)*tmp(:,2:end);

end
