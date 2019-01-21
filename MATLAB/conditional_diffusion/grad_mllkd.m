%% Gradient of negative log-likelihood
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function g_mllkd = grad_mllkd(Fw, J, obs)

g_mllkd = sum( J' * (Fw - obs.data) , 2 ) / obs.std^2;
