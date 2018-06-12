function J = dGds(FEM, s, dt)
%DGDS
%
% Tiangang Cui, 20/May/2014

S       = spdiags(s(:), 0, FEM.DoF, FEM.DoF);
if FEM.ML_flag
    J   = FEM.I + dt*FEM.kappa*FEM.iMLK - dt*FEM.a*FEM.I + 2*dt*FEM.a*S;
else
    J   = FEM.M + dt*FEM.kappa*FEM.K    - dt*FEM.a*FEM.M + 2*dt*FEM.a*FEM.M*S;
end

end