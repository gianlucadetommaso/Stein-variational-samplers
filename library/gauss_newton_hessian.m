%% Gauss-Newton Hessian approximation
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function gnH = gauss_newton_hessian(x, model, prior, obs)

% Calculate Jacobian
J = dFdx(x, model);

% Gauss-Newton Hessian of the negative log-posterior density
gnH = prior.C0i + J' * J / obs.std2;

    
