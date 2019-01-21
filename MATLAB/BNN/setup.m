% setup 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the data file
info.data_base  = './data_sets/data.mat';

%{
info.data_name  = 'naval_propulsion';
% index for inputs and prediction
info.x_indices  = 1:16;   % inputs
info.y_indices  = 18;     % prediction
info.data_size  = 11934;
%}

%{
info.data_name  = 'boston_housing';
% index for inputs and prediction
info.x_indices  = 1:13;   % inputs
info.y_indices  = 14;     % prediction
info.data_size  = 506;
%}

%{
info.data_name  = 'concrete_strength';
% index for inputs and prediction
info.x_indices  = 1:8;   % inputs
info.y_indices  = 9;     % prediction
info.data_size  = 1030;
%}


info.data_name  = 'yacht_hydro';
% index for inputs and prediction
info.x_indices  = 1:6;   % inputs
info.y_indices  = 1;     % prediction
info.data_size  = 308;


info.t_ratio    = 0.8;
info.normal_on  = true;

% number of nodes in internal layers
info.N_int_node = [50, 50];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
data    = load_data(info);

% settup the neural network model
model   = NN_model_setup(info);

% setup the activation function and its derivative 
model.act_func      = @(a) max(0, a);
model.act_func_deri = @(a) double(a > 0);

% hyper-parameters and the default value
model.hyper_on      = false;
model.log_gamma     = 0;
model.log_lambda    = -10;

model.n             = model.N_w + 2;

% parameters for setting up hyper-prior
model.alpha_gamma   = 6;
model.beta_gamma    = 6;
model.alpha_lambda  = 6;
model.beta_lambda   = 6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train the model by computing the MAP
%
theta_init = [rand(model.N_w, 1) ; model.log_gamma; model.log_lambda];

%{
opt_HM = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,... 
    'HessianMultiplyFcn',@(HI,dt) matvec_Fisher(model, data, HI, dt), 'Display','iter', ...
    'MaxIterations', 300);
map_t   = fminunc_2018a(@(t) minus_log_post(model, data, t), theta_init, opt_HM);

RMSE = validate(model, data, map_t, 1);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% in house implementation with Newton-CG

% options for line search
opt_def.line_step       = 1;
opt_def.grad_tol        = 1E-8;
opt_def.step_tol        = 1E-8;
opt_def.func_tol        = 1E-12;
opt_def.max_step        = 200;
opt_def.line_ftol       = 1E-2;
opt_def.line_max_feval  = 10;

% options for Newton-CG
opt_def.CG_restart      = 50;
opt_def.CG_forcing_tol  = 0.1;
opt_def.CG_max_iter     = 200;
opt_def.CG_zero_tol     = 1E-3;


model.map_l = line_search(opt_def, model, data, theta_init);
RMSE = validate(model, data, model.map_l, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



