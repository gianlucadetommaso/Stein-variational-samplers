function [U, s, V, Jv] = svd_explicit_WJ(model, obs, prior, v, tol, nmax)
%SVD_EXPLICIT_WJ
%
% SVD for factorizing the forward model
% requires the HI by runing 
% [~,~,~, HI] = minus_log_post(model, prior, v);
%
% Tiangang Cui, 17/Jan/2014

if nargin == 4
    tol     = 1E-1;
    nmax    = 0;
end

if nargin == 5
    nmax    = 0;
end

[~,~,~,~,HI]= minus_log_post(model, obs, prior, v);

Jx          = explicit_J(model, HI);
if length(obs.std(:)) > 1
    Jx      = scale_rows(Jx, 1./obs.std);
else
    Jx      = Jx/obs.std;
end
Ju          = scale_cols(Jx, HI.dxdu);
Jv          = matvec_prior_Lt(prior, Ju')';

[U,S,V]     = svd(Jv,'econ');

[dS,ind]    = sort(diag(S), 'descend');
i           = dS>=tol;
U           = U(:,ind(i));
V           = V(:,ind(i));
s           = dS(i);


end