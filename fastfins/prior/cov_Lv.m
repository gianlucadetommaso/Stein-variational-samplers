function u = cov_Lv(cov, v)
%COV_LV
%
% v ~ N(0, I)
% u ~ N(0, C)
%
% Tiangang Cui, 17/Jan/2014

switch cov.type
    case {'MRF'}
        u               = zeros(size(v));
        tmp             = scale_rows(v(cov.per,:), cov.sca);
        u(cov.per,:)    = cov.R\(cov.R'\tmp);
    case {'GP'}
        u               = cov.RC'*v;
    case {'KL'}
        u               = cov.RC'*v;
end

end
