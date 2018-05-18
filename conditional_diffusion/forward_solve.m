%% Forward solve
%
% By Gianluca Detommaso -- 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function Fu = forward_solve(x, model)

% Initialise solution and forward map
sol = x(1);
Fu  = [];

for j = 2:model.n
    % Calculate next step
    sol = sol*( 1 + model.beta*( 1-sol^2 ) / ( 1+sol^2 ) * model.h ) + (x(j) - x(j-1));
    
    if mod(j, model.ratio) == 0
        Fu = [Fu; sol]; %#ok<AGROW>
    end
     
end