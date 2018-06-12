% A note for setting up the model problem
%
% Tiangang Cui, 17/Jan/2014
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions for defining log-posterior, log-likelihood, objective function
% used for optimization, and Hessian vector product
% 
% minus_log_post(model, prior, v)
%
% get_map_matlab(model, prior, v)
%
% minus_log_prior(v)
%
% matvec_PPGNH(model, prior, hessinfo, v)
%
% eigen_PPGNH (model, prior, v)
% 
% svd_whitening_F_rand(model, v)
%
% svd_whitening_F     (model, v)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Functions used for defining operation with the prior, tarnsformation
% between parameters
% 
% prior_L_mult(prior, v)
%   transform from whiten parameter v to parameter u, C = L*Lt, this function 
%   gives L*v
%
% prior_Lt_mult(prior, v)
%   C = L*Lt, this function gives Lt*v
%
% pre_process(prior, v, flag)
%   compute the transformation from v to x, and relevant Jacobians
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% model.forward(x); 
%   forward solve
%
% model.adjoint(soln, misfit)
%   given the forward solve solutoin, and data misfit, compute the adjoint
%   gradient
%
% model_func.matvec_GNH(hessinfo, dx); % matvec with GNH for the model
%
% model.matvec_J_left (hessinfo, dy)
%
% model.matvec_J_right(hessinfo, dx)
%
% model.explicit_J(hessinfo)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% These are default settings when the PDE model come with FastFInS is used
%
% model.forward       = @(x)      forward_solve        (mode.FEM, x);
% model.adjoint       = @(s, m)   adjoint_grad         (model.mesh, model.obs, model.FEM, s, m);
% model.matvec_GNH    = @(H, dx)  adjoint_hessmult_NP  (model.mesh, model.obs, model.FEM, H, dx);
% model.matvec_Jty    = @(H, dy)  adjoint_jacmult_left (model.mesh, model.obs, model.FEM, H, dy);
% model.matvec_Jx     = @(H, dx)  adjoint_jacmult_right(model.mesh, model.obs, model.FEM, H, dx);
% model.explicit_J    = @(H)      adjoint_jacobian     (model.mesh, model.obs, model.FEM, H);
% 
% model.PPGNH.tol     = 0.01;
% model.PPGNH.num     = min(model.obs.N_obs, prior.DoF);
% 
% model.SVD.Jm        = model.obs.N_elect;
% model.SVD.Jn        = prior.DoF; 
% model.SVD.num_rand = model.obs.N_obs + 5;
%

