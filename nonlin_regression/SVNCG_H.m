%% Stein Variational Newton-CG with scaled Hessian kernel
%
% By Gianluca Detommaso - 20/10/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function [x, stepsize, timeave] = SVNCG_H(x, stepsize, itermax, model, prior, obs)

% options for Newton-CG
opt_def.CG_restart      = 50;
opt_def.CG_forcing_tol  = 0.1;
opt_def.CG_max_iter     = 200;
opt_def.CG_zero_tol     = 1E-3;

n = model.n;

% Number of particles
N = size(x,2);

% Initialise particle maximum shifts
maxmaxshift_old = inf;

% Initialise average computational time
timeave = 0;

% Initialise H_J matrix
H_J = zeros(n*N);

for k = 1:itermax

    tic;
    
    % Calculate gradient and Gauss-Newton Hessian of the posterior for each particle
    g_mlpt = zeros(model.n, N);
    gnH    = zeros(model.n, model.n, N);
    
    for j = 1:N
        [Fx, J]     = forward_solve(x(:,j), model);
        g_mlpt(:,j) = grad_mlpt(x(:,j), Fx, J, prior, obs);
        gnH(:,:,j)  = prior.C0i + J'*J / obs.std2;   
    end
    
    % Scaled averaging Hessian approximation
    sEH = mean(gnH,3) / n;
    
    % Scaled averaging Hessian approximation
    [kern, g_kern] = kernel_full(x, sEH);
    
    grad = zeros(n, N);
    for i = 1:N       
        % Gradient of the map 
        grad(:,i) = mean( -kern(i,:) .* g_mlpt + g_kern(:,:,i), 2 ); 
    end    
    alpha = newtonCG(x, opt_def, -grad, kern, g_kern, model, prior, obs);     
    alpha = reshape(alpha, n, N);
 
    % Find update directions
    Q = alpha * kern;
    
    % Update particles
    x = x + stepsize * Q;
    
    % Maximum shift over all the particles
    maxmaxshift = max( Q(:) ); 
    fprintf('Maximum shift is %f\n', maxmaxshift)
    
    % Rescale stepsize and reset particles if maximum shift is too large
    if isnan(maxmaxshift) || maxmaxshift > 1e50   
        stepsize = 0.1*stepsize;
        fprintf('Step size too large; scaling it by factor 10.\n epsilon = %f. \n', stepsize), pause(1)
        fprintf('Reset particles... \n'), pause(1)
        x = prior.m0 + prior.C0sqrt*randn(model.n,N); 
    end

    % Update stepsize
    if maxmaxshift >= maxmaxshift_old
        stepsize = 0.9*stepsize;
    elseif abs(maxmaxshift - maxmaxshift_old) < 1e-6
        stepsize = 1.01*stepsize;
    end
    maxmaxshift_old = maxmaxshift;

    % Last iteration
    if k == itermax
       fprintf('Maximum number of iterations has been reached.\n') 
    end
        
    % Update averaged computational time
    timeave = timeave + toc;
end

% Normalise averaged computational time
timeave = timeave / itermax;