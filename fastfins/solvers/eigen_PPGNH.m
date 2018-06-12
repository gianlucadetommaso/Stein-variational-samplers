function [V, d, HI] = eigen_PPGNH(model, obs, prior, v, tol, nmax)
%EIGEN_PPGNH
%
% Eigendecomposition of the prior-preconditioned Hessian
% requires the HI by runing 
% [~,~,~, HI] = minus_log_post(model, prior, v);
%
% tol:       is the truncation tolerance
% max_eigen: is the maximum number of modes solved by eigs
%
% Tiangang Cui, 19/Mar/2014

[~,~,~,~,HI]= minus_log_post(model, obs, prior, v);

opts.issym  = 1;
opts.isreal = 1;

[V, D]      = eigs(@(dv)  matvec_PPGNH(model, obs, prior, HI, dv), prior.DoF,  nmax, 'LA', opts);
d           = diag(D);
i           = d>tol;
V           = V(:,i);
d           = d(i);

end