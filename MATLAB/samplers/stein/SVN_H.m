%% Stein Variational Newton with scaled Hessian kernel
%
% By Gianluca Detommaso - 14/08/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function [x, stepsize, timeave] = SVN_H(x, stepsize, itermax, model, prior, obs)

% Number of particles
N = size(x,2);

% Initialise particle maximum shift
maxmaxshift_old = inf;

% Initialise average computational time
timeave = 0;

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
    sEH = mean(gnH,3) / model.n;
   
    % Initialise matrix of coefficients
    alpha = zeros(model.n, N);
    kern  = zeros(N, N);
    
    for i = 1:N
        
        % Calculate signed difference matrix
        sign_diff = x(:,i) - x;

        % Calculate kernel
        kern(i,:) = exp( -0.5 * sum( sign_diff' * sEH .* sign_diff', 2 ) )';

        % Gradient of kernel
        g_kern = sEH * sign_diff .* kern(i,:);
        
        % Gradient of the map 
        mgrad_J = mean( -kern(i,:) .* g_mlpt + g_kern, 2 ); 
        
        % Hessian of the map
        H_J = mean( permute( repmat( kern(i,:).^2, [model.n 1 model.n]), [1 3 2] ) ...
                             .* gnH , 3 ) + g_kern * g_kern' / N;  

        % Newton direction
        alpha(:,i) = H_J \ mgrad_J;
    end
    
    % Find update directions
    Q = alpha;
    
    % Update particles
    x = x + stepsize*Q;
    
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