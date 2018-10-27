function [x, f] = line_search(opt_def, model, data, x_init)
% LINE_SEARCH    perform line seach, using either Wolfe condition or Armijo
%                backtracking, search direction is given by one of inexact
%                Newton, nonlinear CG or BFGS
%
% Tiangang Cui, 06/August/2018

% initialize
x           = x_init;
[f,g,HI]    = minus_log_post(model, data, x);
step        = opt_def.line_step;
i           = 0;
break_flag  = 0;
f0          = f;

fprintf('Iter\tObjective Fun\tStep Size\t1st Order\tNum CG\tNum Line Search\n');
while 1
    % get the next search direction
    [p, ncg]    = newton_cg(opt_def, model, data, HI, g);

    % Armijo
    [nx,nf,nstep,li]   = armijo_search(opt_def, model, data, step, x, f, g, p);
    step            = min(nstep, 1);
    i               = i+1;
    
    if abs(nf - f)/f0 < opt_def.func_tol
        break_flag = 1;
    end
    
    norm_x          = norm(nx - x);
    if norm_x < opt_def.step_tol
        break_flag = 2;
    end
    
    if i > opt_def.max_step
        break_flag = 4;
    end
    
    x           = nx;
    f           = nf;

    [~,g,HI]    = minus_log_post(model, data, x);
    norm_g      = norm(g);

    if norm_g < opt_def.grad_tol
        break_flag  = 3;
        break;
    end

    fprintf('%4i\t%10.5E\t%10.5E\t%10.5E%10i%10i\n', [i, f, norm_x, norm_g, ncg, li]);
    
    if break_flag > 0
        break;
    end
end
    
switch break_flag
    case {1}
        disp('no improvement in function values, stop')
    case {2}
        disp('no improvement in step size, stop')
    case {3}
        disp('first order optimality met, stop')
    case {4}
        disp('maximum number of step number reached, stop')
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nx, nf, nstep, i] = armijo_search(opt_def, model, data, step, x, f, g, p)
% ARMIJO_SEARCH    performs a line search satisfies the armijo condition
% [next, a] = armijo_search(opt_def, curr)
%
% given a direction of the line search, compute the next point that satify
% the Armijo condition
%
% curr: information at current point, has
%   --    x: current point
%   --    f: function value
%   -- grad: gradient
%   --    p: search direction
%
% find f(x + a*p) <= f(x) + ftol*a*grad_f(x)'*p
%      abs( grad_f(x+a*p)' * p ) <= gtol grad_f(x)'*p
%
% the initial choice of the jump size as 1 for newton and quasi newton,
% there is no maximum jump size, since our problem is propoper.
%
% Tiangang Cui, 15/Mar/2013

f0      = f;    % initial function value
d0      = g'*p; % initial directional derivative
nstep   = step;

if d0 > 0
    disp('Bad search direction');
    nx  = x;
    nf  = f;
    return;
end

golden_ratio    = 0.5*(sqrt(5) + 1);


% start up, assign a initial step size and set the function evalaution
% counter to 1
na  = step;
i   = 1;

%figure
%plot(curr.p)

while 1
    % minus log post value, gradient, and gradient of the likelihood
    % na 
    nf  = minus_log_post_fast(model, data, x + na*p);
    
    if isnan(nf) || isinf(nf)
        na  = na*0.5;
        continue;
    end

    % check for suffcient decrease, if not met
    if nf >= (f0 + opt_def.line_ftol*na*d0)
        if i <= opt_def.line_max_feval
            i   = i+1;
            na  = na*(1-1/golden_ratio);
            continue;
        else
            disp('Max line search iterations reached');
            nx  = x + na*p;
            break;
        end
    end
    % if met
    nx  = x + na*p;
    break;
end

if i == 1
    nstep   = step*2;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, i, pdef] = newton_cg(opt_def, model , data, HI, grad)
% NEWTON_CG_LINE   Conjugate gradient inexact Newton solver, Hx = -g;
%
% delta > 0 gives the Steihaug CG
%
% Tiangang Cui, 06/August/2018

pdef = true;

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
    % Hessmult, apply to the congugate direction
    Hcd = matvec_Fisher(model, data, HI, cd);
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
        r = matvec_Fisher(model, data, HI, x) + grad;
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

