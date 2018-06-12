function u = matvec_prior_Lt(prior, v)
%MATVEC_PRIOR_LT
%
% L'*v
%
% Tiangang Cui, 17/Jan/2014

switch prior.type
    case {'Dist'}
        u = cov_Ltv(prior.cov, v);
    case {'Basis'}
        u = prior.basis'*v;
end

end