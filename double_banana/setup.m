%% Double banana setup
%
% By Gianluca Detommaso -- 14/03/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

%% Model         

% Set length of the state
model.n = 2;
% Length of the forward map
model.m = 1;
% Forward map handle
a       = 1;
b       = 100;
model.F = @(u) log( (a - u(1))^2 + b*(u(2) - u(1)^2)^2 ); 


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
obs.u_true  = rand(model.n);
% Observations
obs.y       = forward_solve(obs.u_true, model) + obs.noise;
