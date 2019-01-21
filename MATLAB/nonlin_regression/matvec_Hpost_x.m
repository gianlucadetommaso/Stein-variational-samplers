%% Hessian-vector multiplication
%
% By Gianluca Detommaso - 17/09/2018
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %

function Hpost_x = matvec_Hpost_x(dx, z, model, prior, obs)

Jx  = 3*model.c(1)*z(1)^2 * dx(1,:) + model.c(2) * dx(2,:); 
JtJx = [ 3*model.c(1)*z(1)^2; model.c(2) ] .* Jx; 

Hpost_x = prior.C0i * dx + JtJx / obs.std2;

end

