%% Forward solve
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function [Fu, J] = forward_solve(x, model)

    Fu = feval(model.F, x);
    
    if nargout > 1
       J = [ 3*model.c(1)*x(1)^2, model.c(2) ]; 
    end
        
     
end