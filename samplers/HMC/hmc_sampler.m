%% Hamiltonian Monte Carlo
%
% By Gianluca Detommaso - 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function out = hmc_sampler(model, prior, obs, hmc)

% Initialise the output chain
out.x = [hmc.init, zeros(model.n, hmc.nsamp-1)];

% Initialise current structure
curr.x  = hmc.init;                                                     % state
curr.Fx = forward_solve(curr.x, model);                                 % forward map
[curr.mlpt, curr.mllkd] = minus_log_post(curr.x, curr.Fx, prior, obs);  % minus log-posterior and -likelihood

% Initialise output
out.mlpt  = [curr.mlpt,  zeros(1, hmc.nsamp-1)];   % negative log-posterior
out.mllkd = [curr.mllkd, zeros(1, hmc.nsamp-1)];   % negative log-likelihood

% Count number of accepted samples 
acc = 0;

for k = 2:hmc.nsamp
    
    % Print iterations
    if mod(k,100) == 0
        fprintf('Iteration %d\n', k)
    end
    
    % Propose candidate sample
    next = propose(curr, model, prior, obs, hmc);
    
    % Acceptance log-ratio
    logratio = -next.ham + next.ham0;
    
    if log(rand) < logratio
        % Accept the candidate state
        out.x(:,k) = next.x;      
        % Increase the counter
        acc        = acc + 1;     
        % Update the current class
        curr       = next;

    else
        % Reject the candidate state
        out.x(:,k) = curr.x;       
    end
    
    % Store minus log-likelihood and log-posterior
    out.mllkd(k) = curr.mllkd;
    out.mlpt(k)  = curr.mlpt;

end

% Calculate rate of acceptance
out.accratio = acc / (hmc.nsamp-1);

end