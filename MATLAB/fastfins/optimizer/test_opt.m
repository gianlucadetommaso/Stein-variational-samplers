% run startup script
startup;

%%%%%%%%%%%%%% defining the forward problem
model_def.problem = 'Laplace'; % or 'Heat'
model_def.test_case = 'EIT_smooth'; % Boundary condition and source term
%model_def.test_case = 'NBC_tarek'; % Boundary condition and source term

% define the mesh size of the forward model
model_def.mesh_size = 50;

% define the type of Hessian used in the Hessmult
model_def.hess_type = 'Full'; % or use 'GN' for Gauss Newton
% after excuting this script, this option can be changed by set
% FEM.hess_flag = 'Full' or 'GN'

%%%%%%%%%%%%%% end


%%%%%%%%%%%%%% defining parameter type
param_def.type = 'Distributed'; % given 'Patition' or 'Bilinear' for affine case
param_def.func = 'log'; % also can use 'erf', if not given, default in no transformation
param_def.log_thres = 0; % this must be set for log transformation

% defining test image type, check "help contents" for the list of choice
% param_def.image_type = 'Prior';
param_def.image_type = 'CF';
param_def.image_base = 2;
param_def.image_range = 2;

%%%%%%%%%%%%%% end

%%%%%%%%%%%%%%  defining prior type
prior_def.type = 'GP';
prior_def.scale = 0.5; % scale
prior_def.power = 1; % power
prior_def.sigma = 1.25; % variance
prior_def.beta = 1; % weighting
% prior bounds
prior_def.use_bounds = false;
prior_def.lb = 0.1;
prior_def.ub = 10;

%%%%%%%%%%%%%% end

dir = [root '/junkyard'];

%%%%%%%%%%%%%% defining the observation observation, usually use type 1, 
% defined by the tensor produdt of corrdinates
obs_def.type = 1; % type 2  for a given region, type 3 for the whole domain
% obs_def.locs defines thetensor grid
obs_def.locs = linspace(0.2,0.6,5);
% signal to noise ratio if this value less than 0, use the prespecified std
obs_def.s2n = 50; 
% obs_def.u_std = 0.1;

%%%%%%%%%%%%%% end

%%%%%%%%%%%%%%  define the data i/o 
output.data_str = [dir '/test_opt_eit.mat'];
%output.data_str = [dir '/test_opt.mat'];
%output.mcmc_str = [dir '/less_10_exp_samples_' num2str(model_def.mesh_size)];
%output.image_str = [root '/param_redu/exp_image_' num2str(model_def.mesh_size)];

%%%%%%%%%%%%%% end

initialize; % initialize the problem setup
plot_scenario; % plot the setup

% MAP estimate
[zmap, xmap] = test_MAP(mesh, mesh_dual, obs, prior, param, FEM);