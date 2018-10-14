%% Stein Variational Newton with scaled Hessian kernel
%
% By *** - 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function [x, stepsize, timeave] = SVNfull_H(x, stepsize, itermax, model, prior, obs)

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
        
        for j = 1:N
            % Hessian of the map
            H_J((i-1)*n+1:i*n, (j-1)*n+1:j*n) = ...
                mean( permute( repmat( kern(i,:) .* kern(j,:), [model.n 1 model.n]), [1 3 2] ) ...
                                 .* gnH , 3 ) + g_kern(:,:,i) * g_kern(:,:,j)' / N;  
        end
        
    end
    
    H_J = H_J + 1e-6*speye(n*N);
    % Search direction
    [L, D] = ldl(H_J);
    diagD = diag(D);
    diagDinv = 1./diagD;
    diagDinv(diagD < 1e-1) = 0;
    
    alpha = L' \ (diagDinv .* (L \ reshape(grad, n*N, 1)));
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