function u = cov_Cv(cov, v)
%COV_CV
%
% Tiangang Cui, 17/Jan/2014

switch cov.type
    case {'MRF'}
        tmp1            = cov.R\(cov.R'\v(cov.per,:));
        tmp2(cov.per,:) = scale_rows(tmp1, cov.sca);
        u               = zeros(size(v));
        tmp3            = scale_rows(tmp2(cov.per,:), cov.sca);
        u   (cov.per,:) = cov.R\(cov.R'\tmp3);
    case {'GP'}
        u               = cov.C*v;
end

end