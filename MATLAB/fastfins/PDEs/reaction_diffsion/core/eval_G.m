function w  = eval_G(FEM, s0, s, dt)
%M_EVAL_F
%
% Tiangang Cui, 20/May/2014

if FEM.ML_flag
    w   = s - s0 + dt*FEM.kappa*(FEM.iMLK*s) - dt*FEM.a*(s.*(1-s));
else    
    w   = FEM.M*s - FEM.M*s0 + dt*FEM.kappa*(FEM.K*s) - dt*FEM.a*(FEM.M*(s.*(1-s)));
end

end