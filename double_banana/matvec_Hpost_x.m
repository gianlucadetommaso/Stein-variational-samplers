%% Hessian-vector multiplication
%
% By Gianluca Detommaso - 17/09/2018
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %

function Hpost_x = matvec_Hpost_x(dx, z, model, prior, obs)

[ ( 2*( z(1) - 1 - 200*z(1)*(z(2) - z(1)^2) ) ) ...
           / ( 1 + z(1)^2 - 2*z(1) + 100*(z(2) - z(1)^2)^2 ), ...
           ( 200*(z(2) - z(1)^2) ) ...
           / ( 1 + z(1)^2 - 2*z(1) + 100*(z(2) - z(1)^2)^2 ) ]; 

Jx  = ( 2*( z(1) - 1 - 200*z(1)*(z(2) - z(1)^2) ) ) ...
           / ( 1 + z(1)^2 - 2*z(1) + 100*(z(2) - z(1)^2)^2 ) * dx(1,:) ...
      + ( 200*(z(2) - z(1)^2) ) ...
           / ( 1 + z(1)^2 - 2*z(1) + 100*(z(2) - z(1)^2)^2 ) * dx(2,:);
JtJx = [ ( 2*( z(1) - 1 - 200*z(1)*(z(2) - z(1)^2) ) ) ...
           / ( 1 + z(1)^2 - 2*z(1) + 100*(z(2) - z(1)^2)^2 ); ...
           ( 200*(z(2) - z(1)^2) ) ...
           / ( 1 + z(1)^2 - 2*z(1) + 100*(z(2) - z(1)^2)^2 ) ] .* Jx; 

Hpost_x = prior.C0i * dx + JtJx / obs.std2;

end

