function v = matvec_prior_invL(prior, u)
%MATVEC_PRIOR_INVL
%
% Whitening transformation
%
% Tiangang Cui, 17/Jan/2014

switch prior.type
    case {'Dist'}
        v = cov_invLu(prior.cov, u);
    case {'Basis'}
        v = prior.basis_w'*u;

end

end