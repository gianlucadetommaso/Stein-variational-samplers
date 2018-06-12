%% Gradient of negative log-likelihood
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function g_mllkd = grad_mllkd(Fx, J, obs)

g_mllkd = sum( J' * (Fx - obs.y) , 2 ) / obs.std2;
