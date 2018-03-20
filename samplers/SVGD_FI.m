%% Stein Variational Gradient Descent with Fisher Information kernel
%
% By Gianluca Detommaso - 15/03/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function x = SVGD_FI(x, stepsize, itermax, timemax, model, prior, obs)

% Initialise clock
tic;

% Number of particles
N = size(x,2);

% Initialise particle maximum shifts
maxshift        = zeros(N,1);
maxmaxshift_old = inf;

for k = 1:itermax
    
    % Stop if over maximum time
    if toc > timemax
        break
    end
    
    % Calculate gradient and Gauss-Newton Hessian of the posterior for each particle
    g_mlpt = zeros(model.n, N);
    gnH    = zeros(model.n, model.n, N);
    
    parfor j = 1:N
        Fu          = forward_solve(x(:,j), model);
        g_mlpt(:,j) = grad_mlpt(x(:,j), Fu, model, prior, obs);
        gnH(:,:,j)  = gauss_newton_hessian(x(:,j), model, prior, obs);
    
    end
    
    % Fisher Information approximation
    FI = mean(gnH,3);
    
    % Copy variable for parfor slicing issues
    x_copy = x;
    
    parfor i = 1:N
        
        % Calculate signed difference matrix
        sign_diff = x(:,i) - x_copy;

        % Calculate kernel
        kern = exp( -0.5 * sum( sign_diff' * FI .* sign_diff', 2 ) )';

        % Gradient of kernel
        g_kern = FI * sign_diff .* kern;
        
        % Calculate the gradient of the pushforward transport map
        mgrad_J = mean( -kern .* g_mlpt + g_kern, 2 ); 
         
        % Search direction
        Q = mgrad_J; 
        
        % Update the particle
        x(:,i) = x(:,i) + stepsize*Q;
        
        % Particle maximum shift
        maxshift(i) = norm(Q, inf);

    end
  
    % Maximum shift over all the particles
    maxmaxshift = max(maxshift);
    fprintf('Maximum shift is %f\n', maxmaxshift)
    
    % Rescale stepsize and reset particles if maximum shift is too large
    if isnan(maxmaxshift) || maxmaxshift > 1e50   
        stepsize = 0.1*stepsize;
        fprintf('Step size too large; scaling it by factor 10.\n epsilon = %f. \n', stepsize), pause(1)
        fprintf('Reset particles... \n'), pause(1)
        x = mvnrnd(prior.m0, prior.C0, N);
    end

    % Update stepsize
    if maxmaxshift >= maxmaxshift_old
        stepsize = 0.9*stepsize;
    else
        stepsize = 1.01*stepsize;
    end
    maxmaxshift_old = maxmaxshift;
    
    % Last iteration
    if k == itermax
       fprintf('Maximum number of iterations has been reached.\n') 
    end
end