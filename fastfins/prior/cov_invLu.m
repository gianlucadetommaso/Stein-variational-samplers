function v = cov_invLu(cov, u)
%COV_INVLU 
%
% whitening transformation
%
% Tiangang Cui, 17/Jan/2014

switch cov.type
    case {'MRF'}
        v = cov.RQ *u;
    case {'GP'}
        v = cov.RC'\u;
end

end