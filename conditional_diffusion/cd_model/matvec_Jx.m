function Jx = matvec_Jx (model, HI, dw)
%MATVEC_JX
%
% Action of the Jacobian on the perturbation of w
%
% Tiangang Cui, 20/Nov/2013

m   = size(dw, 2);
RHS = [zeros(1,m); dw];
tmp = HI.udu\RHS;
Jx  = sqrt(model.dt)*tmp((model.k+1):model.k:end,:);
Jx  = Jx(:);

end
