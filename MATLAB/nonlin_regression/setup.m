%% Non-linear regression setup
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

%% Model         

% Set length of the state
model.n = 2;
% Length of the forward map
model.m = 1;
% Shape coefficients
model.c = randn(1,model.n);
% Forward map handle
model.F = @(x) model.c(1) * x(1)^3 + model.c(2) * x(2);  


%% Prior

% Prior mean
prior.m0      = zeros(model.n,1);
% Prior covariance matrix
prior.C0      = eye(model.n);
% Prior precision matrix
prior.C0i     = prior.C0^(-1);
% Square root of prior covariance matrix
prior.C0sqrt  = real(sqrtm(prior.C0));
% Square root of prior precision matrix
prior.C0isqrt = real(sqrtm(prior.C0i));

%% Observation

% Number of independent observations
obs.nobs    = 1;
% Noise standard deviation
obs.std     = 0.3;
% Nose variance
obs.std2    = obs.std^2;
% Noise
obs.noise   = obs.std*randn(model.m, obs.nobs); 
% Real parameter vector
obs.u_true  = rand(model.n,1);
% Observations
obs.y       = forward_solve(obs.u_true, model) + obs.noise;
