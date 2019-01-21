function u = cov_Ltv(cov, v)
%COV_LTV
%
% Tiangang Cui, 17/Jan/2014

switch cov.type
    case {'MRF'}
        tmp             = cov.R\(cov.R'\v(cov.per,:));
        u(cov.per,:)    = scale_rows(tmp, cov.sca);
    case {'GP'}
        u               = cov.RC*v;
end

end