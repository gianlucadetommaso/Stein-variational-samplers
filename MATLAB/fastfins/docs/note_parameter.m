% A note on the various definitions of the parameters
%
% Tiangang Cui, 17/Jan/2014
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% x:    is defined as the physical parameter that directly used by the PDE
%       model, when the adjoint model of the PDE is evaluated, it evaluates
%       the adjoint derivative w.r.t. x
%       x is given by: x = f(u).
%
% u:    is defined as the unknown parameter that has a Gaussian prior 
%       N(m, C)
%
% v:    is the compuational parameter that a MCMC algorithm or an optimizer 
%       deal with, it has prior N(0,I). It can also be defined on some
%       subspace.
%
% Case 1: whitening
%
% v:    is the parameter after the whitening transformation applied to u, 
%       and shifted with zero mean, so v has prior N(0, I)
%       v = inv_sqrt(C) * (u - m)
%       u =     sqrt(C) *  v + m
%
% Case 2: KL
%
% v:    is weights associated with the KL basis, and has prior N(0,I)
%       Given   KL.P*(KL.S^2)*KL.P' = C
%       and     basis = KL.P*KL.S, basis_w = KL.P*inv(KL.S)
%
%       v = basis_w'*(u - m), v ~ N(0, I)
%       u = basis*v + m
%
% Case 3: Redeuced subspace
%
% v:    is the weights associated with the basis of the likelihood-informed 
%       subspace, in this case, the reduced basis is orthonormal, and defined 
%       in the space associated with prior N(0, I), thus v has prior N(0, I_r).
%
%       We first define the transformation from v to v', where v' is defined 
%       on the full space. 
%
%       Let     C = L*L', and P is the LIS basis for parameter with N(0, I)
%       and     basis = L*P and basis_w = inv(L')*P
%
%       v' = P*v            P is orthonormal
%       u  = basis*v + m
%       
%       v  = basis_w*(u - m)
%
%       In some cases, we need define a reference point v_ref within the LIS
%       for running MCMC around it. two possibilities here:
%       1) given a reference point in u  ~ N(m, C)
%           v_ref = basis_w*(u_ref - m)
%       2) given a reference point in v' ~ N(0, C)
%           v_ref = P'*v
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%