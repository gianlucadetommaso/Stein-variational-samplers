%% Plot posterior contour
%
% By Gianluca Detommaso - 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function post = post2contour(x, model, prior, obs)
    
    post = zeros(size(x, 2), 1);
    
    for j = 1:size(x, 2)
        
        % Minus log prior
        mlprr = 0.5*(x(:,j) - prior.m0)' * prior.C0i * (x(:,j) - prior.m0);
        
        % Forward operator
        Fu = forward_solve(x(:,j), model);
 
        % Minus log likelihood
        misfit = (obs.y - Fu) / obs.std;
        mllkd  = 0.5*sum(misfit(:).^2);
        
        % Evaluate the post
        post(j) = exp( -(mllkd + mlprr) );
        
    end

end