function vmap = get_map_matlab(model, obs, prior, v_init)
%GET_MAP_MATLAB   
%
% Runs optimization algorithms to get the MAP estitimate
%
% Tiangang Cui, 04/May/2012


% full hessian, with log transformation
opt_HM = optimset('GradObj','on','Hessian','on','Display','iter',...
    'MaxIter',100,'largescale','on','DerivativeCheck','off',...
    'HessMult',@(HI,v) matvec_hessian(model, obs, prior, HI, v));

if ~isempty(strfind('2016', '2016'))
    vmap   = fminunc_2016b(@(v) obj(model, obs, prior, v), v_init, opt_HM);
else
    vmap   = fminunc_2014b(@(v) obj(model, obs, prior, v), v_init, opt_HM);
end

%{
opt_QN = optimset('GradObj','on','Display','iter',...
    'MaxIter',100,'largescale','off','DerivativeCheck','off',...
    'Algorithm', 'quasi-newton');

vmap   = fminunc(@(v) obj(model, obs, prior, v), v_init, opt_QN);

%}
%{
warning('error', 'MATLAB:singularMatrix');
warning('error', 'MATLAB:nearlySingularMatrix');
while 1
    try
        vmap   = fminunc(@(v) obj(model, obs, prior, v), v_init, opt_HM);
        break;
    catch 
        disp('bad run');
        v_init = randn(prior.DoF, 1);
    end
end
%}

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f, g, hessinfo] = obj(model, obs, prior, v)
%MINUS_LOG_POST_MAP
% 
% Compute ths log posterior for optimization
%
% Tiangang Cui, 23/Mar/2013

[f, dummy, g, dummy, hessinfo] = minus_log_post(model, obs, prior, v);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function w = matvec_hessian(model, obs, prior, HI, dv)

w = matvec_PPGNH(model, obs, prior, HI, dv) + dv;

end
