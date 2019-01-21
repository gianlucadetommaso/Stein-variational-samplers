%% Gradient of negative log-posterior
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function g_mlpt = grad_mlpt(w, Fw, J, obs)

g_mlpt = w + grad_mllkd(Fw, J, obs); 
