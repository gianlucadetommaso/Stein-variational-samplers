function [x, i, pdef] = newton_cg(opt_def, curr)
% NEWTON_CG_LINE   Conjugate gradient inexact Newton solver, with BFGS
%                  preconditioner
%
% delta > 0 gives the Steihaug CG
%
% Tiangang Cui, 16/Mar/2013

% trust region info

pdef = true;

x = zeros(opt_def.np,1);
r = curr.grad;

% congugate direction
cd = -r;
% r'Pr
top = r'*r;

forcing = min(opt_def.CG_forcing_tol, sqrt(curr.n_grad));
forcing = max(0.1, forcing)*curr.n_grad;
%sqrt(top)

% start
i = 0;
while 1
    % Hessmult, apply to the congugate direction
    Hcd = opt_def.hessmult(curr.hessinfo, cd);
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
        r = curr.grad + opt_def.hessmult(curr.hessinfo, x);
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
