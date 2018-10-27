function map = get_map(model, data, theta_init)
%GET_MAP_MATLAB   
%
% Runs optimization algorithms to get the MAP estitimate
%
% Tiangang Cui, 06/August/2018

% full hessian, with log transformation
opt_HM = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,... 
    'HessianMultiplyFcn',@(HI,dt) matvec_Fisher(model, HI, dt), 'Display','iter', ...
    'MaxIterations', 100);

map   = fminunc_2018a(@(t) obj(model, data, t), theta_init, opt_HM);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f, g, hessinfo] = obj(model, data, t)

[f, ~, g, ~, hessinfo] = minus_log_post(model, data, t);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

