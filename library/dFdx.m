%% Jacobian of the forward operator
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function J = dFdx(x, model)
   
    % Initialise the Jacobian of the forward operator
    J = zeros(model.m, model.n);
    
    %For each component, use a central finite difference approximation
    h = 1e-2;
    
    for j = 1:model.n
        xplus     = x;
        xplus(j)  = x(j) + 0.5*h; 
        xminus    = x;
        xminus(j) = x(j) - 0.5*h;
        
        % Approximate partial derivative
        J(:,j)  = ( forward_solve(xplus, model) - forward_solve(xminus, model) ) / h;

    end
     
end