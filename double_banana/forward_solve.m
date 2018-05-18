%% Forward solve
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function Fu = forward_solve(x, model)

    Fu = feval(model.F, x);
     
end