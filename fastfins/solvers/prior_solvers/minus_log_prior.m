function [mlp, grad_p] = minus_log_prior( v )
%MINUS_LOG_PRIOR   
%
% Computes the prior for the diagonalized parameter
%
% Tiangang Cui, 17/Jan/2014

grad_p = v;
mlp    = 0.5*(v'*grad_p);

end