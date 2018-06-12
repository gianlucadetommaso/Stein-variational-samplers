function reduced = basis_LIS(prior, P)
%BASIS_LIS
%
% P:     new basis
% S:     singular values associated with P
% v_0:   reference variable, without prior mean
%
% Tiangang Cui, 20/Oct/2012

reduced         = prior;

reduced.P       = P;
reduced.basis   = matvec_prior_L    (prior, P); % prior_L_mult  (prior, P);
reduced.basis_w = matvec_prior_invLt(prior, P); % prior_Lit_mult(prior, P);

reduced.type    = 'Basis';
reduced.DoF     = size(P,2);
reduced.note    = 'LIS';
end

