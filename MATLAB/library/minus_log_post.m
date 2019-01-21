%% Negative log-posterior
%
% By Gianluca Detommaso - 5/06/2017
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function [mlpt, mllkd] = minus_log_post(u, Fu, prior, obs)

% Evaluate the negative log-likelihood
misfit = obs.y - Fu;
mllkd  = 0.5 * sum(misfit(:).^2) / obs.std2;

% Evaluate the minus log-prior
mlpr = 0.5 * (u - prior.m0)' * prior.C0i * (u - prior.m0);

% Evaluate the minus log-posterior
mlpt = mlpr + mllkd;

end