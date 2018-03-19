%% Stein Variational Quasi-Newton with Isotropic Gaussian kernel
%
% By Gianluca Detommaso -- 15/03/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function x = SVQN_I(N, stepsize, timemax, model, prior, obs)

% Initialise clock
tic;

% Initial particle configurations
x = prior.m0 + prior.C0sqrt*randn(model.n,N);

% Initialise particle maximum shifts
maxshift        = zeros(N,1);
maxmaxshift_old = inf;

while toc <= timemax
    
    if N > 1    % If more than one particle

        % Calculate squared distance between particles
        dist2 = pdist2(x',x','squaredeuclidean');

        % Calculate the squared median 
        med2 = median( dist2(dist2~=0) )^2;

        % Set up the kernel length
        h_inv = log(N)/med2;

    else
        % Calculate the squared elements of the distance
        dist2 = 0;

        % Set up the kernel length
        h_inv = 1;
    end 
    
    % Calculate gradient and Gauss-Newton Hessian of the posterior for each particle
    g_mlpt = zeros(model.n, N);
    
    parfor j = 1:N
        Fu          = forward_solve(x(:,j), model);
        g_mlpt(:,j) = grad_mlpt(x(:,j), Fu, model, prior, obs);
        gnH(:,:,j)  = gauss_newton_hessian(x(:,j), model, prior, obs);
    end
    
    % Fisher information approximation
    FI = mean(gnH, 3);
    
    % Calculate kernel
    kern = exp(-h_inv*dist2);
    
    % Copy variable for parfor slicing issues
    x_copy = x;
    
    parfor i = 1:N
        
        % Calculate signed difference matrix
        sign_diff = x(:,i) - x_copy;

        % Gradient of kernel
        g_kern = 2*h_inv*kern(i,:) .* sign_diff;
        
        % Calculate the gradient of the pushforward transport map
        mgrad_J = mean( -kern(i,:) .* g_mlpt + g_kern, 2 );  
                
        % Hessian of the map
        H_J = mean( permute( repmat( kern(i,:), [model.n 1 model.n]), [1 3 2] ) .* gnH , 3 ) ...
                + FI*mean(kern(i,:));
          
        % Search direction
        Q = H_J \ mgrad_J; 
        
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

end
