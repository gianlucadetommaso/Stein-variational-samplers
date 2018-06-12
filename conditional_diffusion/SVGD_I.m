%% Stein Variational Gradient Descent with Isotropic kernel
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function [w, stepsize, timeave] = SVGD_I(w, stepsize, itermax, model, obs)

% Number of particles
npart = size(w,2);

% Initialise particle maximum shifts
maxshift        = zeros(npart,1);
maxmaxshift_old = inf;

% Initialise average computational time
timeave = 0;

for k = 1:itermax
    
    tic;

    if npart > 1    % If more than one particle

        % Calculate squared distance between particles
        dist2 = pdist2(w',w','squaredeuclidean');

        % Calculate the squared median 
        med2 = median( dist2(dist2~=0) );

        % Set up the kernel length
        h_inv = log(npart)/med2;

    else
        % Calculate the squared elements of the distance
        dist2 = 0;

        % Set up the kernel length
        h_inv = 1;
    end 
    
    % Calculate gradient of the posterior for each particle
    g_mlpt = zeros(model.N, npart);
    
    for j = 1:npart
        soln        = forward_solve(model, w(:,j));
        J           = explicit_J(model, soln);
        g_mlpt(:,j) = grad_mlpt(w(:,j), soln.d, J, obs);
    end
    
    % Calculate kernel
    kern = exp(-h_inv*dist2);
    
    % Copy variable for parfor slicing issues
    w_copy = w;
    
    parfor i = 1:npart
        
        % Calculate signed difference matrix
        sign_diff = w(:,i) - w_copy;

        % Gradient of kernel
        g_kern = 2*h_inv*kern(i,:) .* sign_diff;
        
        % Calculate the gradient of the pushforward transport map
        mgrad_J = mean( -kern(i,:) .* g_mlpt + g_kern, 2 );  
         
        % Search direction
        Q = mgrad_J; 
        
        % Update the particle
        w(:,i) = w(:,i) + stepsize*Q;
        
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
        w = randn(model.n,npart); 
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