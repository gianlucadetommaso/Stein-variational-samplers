function v = matvec_prior_invLt(prior, u)
%MATVEC_PRIOR_INVLT
%
% inv(L')*u
%
% Tiangang Cui, 17/Jan/2014

switch prior.type
    case {'Dist'}
        v = cov_invLtu(prior.cov, u);
    case {'Basis'}
        disp('Not well-defined');
end

end