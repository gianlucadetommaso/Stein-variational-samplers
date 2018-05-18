%% Hamiltonian proposal
%
% By Gianluca Detommaso - 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function next = propose(curr, model, prior, obs, hmc)

% Set initial Hamiltonian
mom0      = hmc.Msqrt*randn(model.n,1);
kin0      = 0.5 * mom0' * hmc.Mi * mom0;
next.ham0 = curr.mlpt + kin0;

% Initialise position and momentum
next.x  = curr.x;
mom     = mom0;
% Make an initial half step for the momentum
next.Fx = curr.Fx;
mom     = mom - 0.5 * hmc.eps * grad_mlpt(next.x, next.Fx, model, prior, obs);

% Alternate full steps for position and momentum
for j = 1:hmc.nsteps
    % Full position step
    next.x = next.x + hmc.eps * hmc.Mi * mom;                            
    if j ~= hmc.nsteps
        next.Fx = forward_solve(next.x, model);
        % Full momentum step
        mom = mom - hmc.eps * grad_mlpt(next.x, next.Fx, model, prior, obs);   
    end
end

% Make a final half step for the momentum
mom = mom - 0.5 * hmc.eps * grad_mlpt(next.x, next.Fx, model, prior, obs);
% Negate the momentum to make the proposal symmetric
mom = -mom;

% Evaluate forward operator
next.Fx = forward_solve(next.x, model);

% Kinetic energy at the end of the trajectory
kin  = 0.5 * mom' * hmc.Mi * mom;

% Calculate negative log-posterior and log-likelihood
[next.mlpt, next.mllkd] = minus_log_post(next.x, next.Fx, prior, obs);

% Hamiltionian
next.ham = kin + next.mlpt;