function u = matvec_prior_L(prior, v)
%MATVEC_PRIOR_L
% 
% L*v
%
% v ~ N(0, I)
% u ~ N(0, C)
%
% Tiangang Cui, 17/Jan/2014

switch prior.type
    case {'Dist'}
        u = cov_Lv(prior.cov, v);
    case {'Basis'}
        u = prior.basis*v;
    case {'KL'}
        u = prior.corr.sigma*v;
end

end
