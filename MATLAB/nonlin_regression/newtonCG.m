function [x, i, pdef] = newtonCG(particles, opt_def, grad, kern, g_kern, model, prior, obs)
% NEWTON_CG_LINE   Conjugate gradient inexact Newton solver, Hx = -g;
%
% delta > 0 gives the Steihaug CG
%
% Tiangang Cui, 06/August/2018

pdef = true;

grad = grad(:);

x       = zeros(size(grad));
r       = grad;

% congugate direction
cd = -r;
% r'Pr
top = r'*r;

norm_g = norm(grad);
%forcing = min(opt_def.CG_forcing_tol, sqrt(norm_g)) * sqrt(norm_g);
forcing = opt_def.CG_forcing_tol * norm_g;
%sqrt(top)

% start
i = 0;
while 1
    % Hessmult, apply to the conjugate direction
    Hcd = matvec_HJ_x(cd, particles, kern, g_kern, model, prior, obs);
    % curvature
    bottom = cd'*Hcd;

    % negative curvature, use the inexact Newton trick
    if bottom/(cd'*cd) < opt_def.CG_zero_tol
    % if bottom < 0
        pdef = false;
        if i == 0
            x = cd;
        else
            alpha = top/bottom;
            x = x + alpha*cd;
        end
        break;
    end
    
    % update x and residual
    alpha = top/bottom;
    x = x + alpha*cd;
    r = r + alpha*Hcd;
    
    i = i + 1;

    % recompute residual after certain number of iterations
    if mod(i, opt_def.CG_restart) == 0
        r = matvec_HJ_x(x, particles, kern, g_kern, model, prior, obs) + grad;
    end
    
    % check residual
    if norm(r) < forcing
        break;
    end
    
    % check the iteration count
    if i >= opt_def.CG_max_iter
        disp('Maximum iteration reached, terminating CG')
        break;
    end
    
    ntop = r'*r;
    % update conjugate direction
    cd = - r + (ntop/top)*cd;
    
    % next iteration
    top = ntop;
end

end

