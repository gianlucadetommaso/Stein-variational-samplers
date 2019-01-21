% This script demonstrates the construction of the likelihood informed 
% subspace (LIS). Then subspace MCMC is simulated within the LIS.
% Adaptive MALA is used for both the LIS construction and the subspace MCMC
% Tiangang Cui, 17/May/2014

mcmc_def.init               = vmap;     % MAP estimate as the initial guess
mcmc_def.minus_log_post     = @(v) minus_log_post(model, obs, prior, v);
mcmc_def.use_curvature      = false;
mcmc_def.ref_type           = 'init';

redu_def.init               = vmap;     % MAP estimate as the initial guess
redu_def.gsvd_Nmax          = 10;      % max number of iterations for constructing LIS 

redu_def.method             = 'Eig';    % using eigendecompositoin 
redu_def.PPGNH              = @(v, tol, Nmax) eigen_PPJtJ(model, obs, prior, v, tol, Nmax);
redu_def.eigen_tol          = 1E-1;
redu_def.eigen_Nmax         = obs.Ndata;    % maximumn number of eigenvector
redu_def.gsvd_trunc_tol     = sqrt(1E-1);   % threshold for truncating the global SVD
redu_def.gsvd_conv_tol      = 1E-10;        % convergence threshold

%{
redu_def.method             = 'Eig';    % using randomized SVD
redu_def.eigen_tol          = 1E-1;
redu_def.eigen_Nmax         = obs.N_obs;    % maximumn number of eigenvector
redu_def.SVD                = @(v) svd_rand_WJ(model, obs, prior, v);
%}

tic;
dimredu             = dili_lis(mcmc_def, redu_def);
toc

param_redu          = dimredu.param_redu;

mcmc_def.init               = vmap;     % MAP estimate as the initial guess
mcmc_def.minus_log_post     = @(v) minus_log_post(model, obs, prior, v);
mcmc_def.use_curvature      = false;
mcmc_def.ref_type           = 'init';
mcmc_def.proposal           = 'MALA';
mcmc_def.save_all_flag      = true;    % save all the parameters
mcmc_def.projection         = param_redu.P;      % projection matrix for the marginal MCMC history
mcmc_def.minus_log_post     = @(v) minus_log_post(model, obs, prior, v);
mcmc_def.nstep              = 1E6;

tic
out_dili                    = dili_mcmc(mcmc_def, param_redu);
toc