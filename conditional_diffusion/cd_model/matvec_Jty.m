function Jty = matvec_Jty(model, HI, dy)
%MATVEC_JTY
%
% Action of the Jacobian transpose on a vector from the model output space
%
% Tiangang Cui, 20/Nov/2013

m   = size(dy, 2);
RHS = zeros(model.N+1, m);
RHS((model.k+1):model.k:end,:) = dy;

tmp = HI.udu'\RHS;
Jty = sqrt(model.dt)*tmp(2:end,:);
    
end

