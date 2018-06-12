function v = cov_invLtu(cov, u)
%COV_INVLTU
%
% Tiangang Cui, 17/Jan/2014

switch cov.type
    case {'MRF'}
        v = cov.RQ'*u;
    case {'GP'}
        v = cov.RC \u;
end

end