%% Gradient of minus log-posterior
%
% By Gianluca Detommaso -- 15/03/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function g_mlpt = grad_mlpt(x, Fx, model, prior, obs)

g_mlpt = prior.C0i*(x - prior.m0) + grad_mllkd(x, Fx, model, obs); 
