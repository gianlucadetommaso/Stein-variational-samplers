function [V, d] = eigen_PPJtJ(model, obs, v, tol, nmax)
%EIGEN_PPJtJ
%
% Eigendecomposition of the prior-preconditioned Hessian, given the
% expilcit Jacobian
%
% Requires the HI by runing 
% [~,~,~, HI] = minus_log_post(model, prior, v);
%
% tol:       is the truncation tolerance
% max_eigen: is the maximum number of modes solved by eigs
%
% Tiangang Cui, 19/Mar/2014

if nargin == 4
    tol     = 1E-2;
    nmax    = 0;
end

if nargin == 5
    nmax    = 0;
end

[~,~,~,~,HI]= minus_log_post(model, obs, v);

Jx          = explicit_J(model, HI);
if length(obs.std(:)) > 1
    Jx      = scale_rows(Jx, 1./obs.std);
else
    Jx      = Jx/obs.std;
end
Ju          = scale_cols(Jx, HI.dxdu);
Jv          = matvec_prior_Lt(prior, Ju')';
H           = Jv'*Jv;

[V, D]      = eig(H);
[d,ind]     = sort(diag(D), 'descend');
i           = d>=tol;
V           = V(:,ind(i));
d           = d(i);

end