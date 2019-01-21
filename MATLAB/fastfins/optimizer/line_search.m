function map = line_search(opt_def)
% LINE_SEARCH    perform line seach, using either Wolfe condition or Armijo
%                backtracking, search direction is given by one of inexact
%                Newton, nonlinear CG or BFGS
%
% Tiangang Cui, 16/Mar/2013

% opt_def

% initialize
curr.x = opt_def.init;
[curr.f, curr.grad, curr.hessinfo] = opt_def.func(curr.x);
curr.n_grad = norm(curr.grad);

f0 = curr.f;

step = opt_def.line_step;

if strcmp(opt_def.solver, 'BFGS')
    Y = zeros(opt_def.np, opt_def.bfgs_max_num);
    S = zeros(opt_def.np, opt_def.bfgs_max_num);
    j = 0;
else
    Y = [];
    S = [];
end

i = 0;
while 1
    % get the next search direction
    switch opt_def.solver
        case {'Newton'}
            [curr.p, iter] = newton_cg(opt_def, curr);
        case {'NCG'}
            % preconditioned conjugate gradient
        otherwise
            % preconditioend BFGS
            curr.p = -two_loops(Y, S, j, curr.grad, 'iH');
            iter = 0;
    end
    
    % Armijo
    [next, step] = armijo_search(opt_def, curr, step);
    step = min(step, 1);
    % adjoint gradient
    [next.f, next.grad, next.hessinfo] = opt_def.func(next.x);
    next.n_grad = norm(next.grad);
    
    % iteration
    i = i+1;
    
    info = iter_info(opt_def, f0, curr, next, i, iter);
    if info > 0
        map = next.x;
        return;
    end
    
    if strcmp(opt_def.solver, 'BFGS')
        % update Y and S
        y = next.grad - curr.grad;
        s = next.x - curr.x;
            
        if y'*s > 1e-5
            j = j+1;
            n = min(j, opt_def.bfgs_max_num);
            if n > 1
                Y(:,2:n) = Y(:,1:(n-1));
                S(:,2:n) = S(:,1:(n-1));
            end
            Y(:,1) = y;
            S(:,1) = s;
        end
    end
    
    % swap state
    curr = next;
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [next, nstep] = armijo_search(opt_def, curr, step)
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


% initial function value
f0 = curr.f;
% initial directional derivative
d0 = curr.grad'*curr.p;

nstep = step;

if d0 > 0
    disp('Bad search direction');
    next = curr;
    return;
end

golden_ratio = 0.5*(sqrt(5) + 1);

% start up, assign a initial step size and set the function evalaution
% counter to 1
na = step;
i = 1;

%figure
%plot(curr.p)

while 1
    % minus log post value, gradient, and gradient of the likelihood
    % na 
    nf = opt_def.func(curr.x + na*curr.p);
    
    % check for suffcient decrease, if not met
    if nf >= (f0 + opt_def.line_ftol*na*d0)
        if i <= opt_def.line_max_feval
            i = i+1;
            na = na*(1-1/golden_ratio);
            continue;
        else
            disp('Max line search iterations reached');
            next = curr;
            break;
        end
    end
    % if met
    next.x = curr.x + na*curr.p;
    next.f = nf;
    break;
end

if i == 1
    nstep = step*2;
end

end
