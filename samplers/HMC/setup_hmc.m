%% Set up Hamiltonian Monte Carlo
%
% By Gianluca Detommaso - 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function hmc = setup_hmc(model, prior)

% Set up total number of samples
hmc.nsamp = 1e4;

% Initial state
hmc.init = prior.m0 + prior.C0isqrt*randn(model.n,1);

% Proposal setup
hmc.nsteps     = 1e1;                   % Number of steps in the Hamiltonian dynamics
hmc.eps        = 1e-2;                  % Stepsize
hmc.M          = eye(model.n);          % Kinetic mass matrix
hmc.Mi         = inv(hmc.M);
hmc.Msqrt      = real(sqrtm(hmc.M));
