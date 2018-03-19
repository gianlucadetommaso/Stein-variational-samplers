%% Gradient of minus log-likelihood
%
% By Gianluca Detommaso -- 15/03/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function g_mllkd = grad_mllkd(x, Fx, model, obs)

g_mllkd = sum( dFdx(x, model)' * (Fx - obs.y) , 2 ) / obs.std2;
