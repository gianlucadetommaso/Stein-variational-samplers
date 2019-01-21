function [V, d] = cov_eig(cov)
%COV_EIG
%
% Tiangang Cui, 17/Jan/2014

switch cov.type
    case{'GP'}
        [V, D]      = eig(cov.C);
    case{'MRF'}
        opts.issym  = 1;
        opts.isreal = 1;
        [V, D]      = eigs(@(dv) cov_Cv(cov, dv), size(cov.Q,1), size(cov.Q,1)-1, 'LA', opts);
    case{'KL'}
        [V, D]      = eig(cov.C);
end

[d, ind]            = sort( diag(D), 'descend' );
 V                  = V(:, ind);

end
