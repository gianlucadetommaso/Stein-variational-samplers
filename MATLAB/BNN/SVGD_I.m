%% Stein Variational Gradient Descent with Isotropic kernel
%
% By Gianluca Detommaso - 08/08/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function [x, stepsize, timeave] = SVGD_I(x, stepsize, itermax, model, data)

% Number of particles
N = size(x,2);

% Initialise particle maximum shifts
maxshift        = zeros(N,1);
maxmaxshift_old = inf;

% Initialise average computational time
timeave = 0;

g_mlpt = zeros(model.n, N);
HI    = cell(N,1);

for k = 1:itermax

    tic;
    
    % Calculate 
    for j = 1:N
        [~, g_mlpt(:,j), HI{j}] = minus_log_post(model, data, x(:,j));
    end
       
    % Scaled averaging Hessian approximation
    [kern, g_kern] = I_kernel(x);
    
    for i = 1:N       
        % Gradient of the map 
        grad = mean( -kern(i,:) .* g_mlpt + g_kern(:,:,i), 2 ); 
        
        % Update the particle
        x(:,i) = x(:,i) + stepsize * grad;
          
        % Particle maximum shift
        maxshift(i) = norm(grad, inf);
    end
  
    % Maximum shift over all the particles
    maxmaxshift = max(maxshift);
    fprintf('Maximum shift is %f\n', maxmaxshift)
    
    % Rescale stepsize and reset particles if maximum shift is too large
    if isnan(maxmaxshift) || maxmaxshift > 1e50   
        stepsize = 0.1*stepsize;
        fprintf('Step size too large; scaling it by factor 10.\n epsilon = %f. \n', stepsize), pause(1)
        fprintf('Reset particles... \n'), pause(1)
        x = model.map_l + randn(model.n,N); 
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