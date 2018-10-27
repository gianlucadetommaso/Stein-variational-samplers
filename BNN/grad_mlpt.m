%% Gradient of negative log-posterior
%
% By Gianluca Detommaso -- 08/08/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function g_mlpt = grad_mlpt(x, model, obs)

n    = size(x,1);
ncol = size(x,2);

g_mlpt = zeros(n,ncol);

parfor j = 1:ncol
    xj = x(:,j);
    g_mlpt(:,j) = xj + sum( matvec_Jty(xj, forward_solve(xj, model) - obs.y), 2) / obs.std2;
end
