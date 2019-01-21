%% Stein Variational Gradient Descent with Isotropic kernel
%
% By *** -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function [x, stepsize, timeave] = SVGD_I(x, stepsize, itermax, model, prior, obs)

% Number of particles
N = size(x,2);

% Initialise particle maximum shifts
maxshift        = zeros(N,1);
maxmaxshift_old = inf;

% Initialise average computational time
timeave = 0;

for k = 1:itermax
    
    tic;

    if N > 1    % If more than one particle

        % Calculate squared distance between particles
        dist2 = pdist2(x',x','squaredeuclidean');

        % Calculate the squared median 
        med2 = median( dist2(dist2~=0) );

        % Set up the kernel length
        h_inv = log(N)/med2;

    else
        % Calculate the squared elements of the distance
        dist2 = 0;

        % Set up the kernel length
        h_inv = 1;
    end 
    
    % Calculate gradient of the posterior for each particle
    g_mlpt = zeros(model.n, N);
    
    for j = 1:N
        [Fx, J]     = forward_solve(x(:,j), model);
        g_mlpt(:,j) = grad_mlpt(x(:,j), Fx, J, prior, obs);
    end
    
    % Calculate kernel
    kern = exp(-h_inv*dist2);
    
    for i = 1:N
        
        % Calculate signed difference matrix
        sign_diff = x(:,i) - x;

        % Gradient of kernel
        g_kern = 2*h_inv*kern(i,:) .* sign_diff;
        
        % Calculate the gradient of the pushforward transport map
        mgrad_J = mean( -kern(i,:) .* g_mlpt + g_kern, 2 );  
         
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