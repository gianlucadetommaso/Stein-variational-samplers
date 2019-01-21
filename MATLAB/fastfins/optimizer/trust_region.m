function map = trust_region(opt_def)
% TRUST_REGION   Subspace inexact Newton trust region solver
%
% Tiangang Cui, 16/Mar/2013
% opt_def
% initialize
curr.x = opt_def.init;
[curr.f, curr.grad, curr.hessinfo] = opt_def.func(curr.x);
curr.n_grad = norm(curr.grad);

f0 = curr.f;

if strcmp(opt_def.solver, 'BFGS')
    Y = zeros(opt_def.np, opt_def.bfgs_max_num);
    S = zeros(opt_def.np, opt_def.bfgs_max_num);
    j = 0;
else
    Y = [];
    S = [];
end

% initial trust region
delta = 0.05*opt_def.TR_radius;

i = 0;
while 1
    % subspace search
    switch opt_def.solver
        case {'Newton'}
            [p, iter, pdef] = newton_cg(opt_def, curr);
            %if p'*curr.grad>0
            %    disp('bad');
            %    return;
            %end
            % subspace is g and p
            if pdef || iter > 1
                % solve the 2d problem
                u = -curr.grad/curr.n_grad;
                v = p - u*(u'*p);
                nv = norm(v);
                if nv > 1e-10
                    v = v/nv;
                    U = [u, v];
                    B = U'*opt_def.hessmult(curr.hessinfo, U);
                    a = U'*curr.grad;
                    [s, quad_decrease] = two_space_sub(a, B, delta);
                    jump = U*s;
                else
                    d = -(delta/curr.n_grad)*curr.grad;
                    a = 0.5*d'*opt_def.hessmult(curr.hessinfo, d);
                    b = curr.grad'*d;
                    [jump, quad_decrease] = cauchy(a, b, d);
                end
            else
                d = -(delta/curr.n_grad)*curr.grad;
                a = 0.5*d'*opt_def.hessmult(curr.hessinfo, d);
                b = curr.grad'*d;
                [jump, quad_decrease] = cauchy(a, b, d);
            end
        otherwise
            p = -two_loops(Y, S, j, curr.grad, 'iH');
            iter = 0;
            
            u = -curr.grad/curr.n_grad;
            v = p - u*(u'*p);
            nv = norm(v);
            if nv > 1e-5
                v = v/nv;
                U = [u, v];
                tmp1 = two_loops(Y, S, j, U(:,1), 'H');
                tmp2 = two_loops(Y, S, j, U(:,2), 'H');
                B = U'*[tmp1, tmp2];
                a = U'*curr.grad;
                [s, quad_decrease] = two_space_sub(a, B, delta);
                jump = U*s;
            else
                d = -(delta/curr.n_grad)*curr.grad;
                a = 0.5*d'*two_loops(Y, S, j, d, 'H');
                b = curr.grad'*d;
                [jump, quad_decrease] = cauchy(a, b, d);
            end
    end
    
    next.x = curr.x + jump;
    % get to the next point
    [next.f, next.grad, next.hessinfo] = opt_def.func(next.x);
    next.n_grad = norm(next.grad);
    
    i = i + 1;
    info = iter_info(opt_def, f0, curr, next, i, iter);
    if info > 0
        map = next.x;
        return;
    end
    
    % evaluate the gain
    rho = (next.f - curr.f)/quad_decrease;
    
    if rho < 0.25
        delta = 0.25*delta;
    else
        if rho > 0.75 && norm(jump) > (delta - 1e-2)
            delta = min(opt_def.TR_radius, 2*delta);
        end
    end
    
    % update
    if rho > 0.1
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
    
    if rho < 1E-10
        delta = delta*0.1;
    end
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [s, quad_decrease] = two_space_sub(a, B, delta)

% min  g'Vs + 0.5*s'*V'*H*V*s
% s.t. s'V'Vs <= delta^2
%
% min a'*s + 0.5 s'*B*s, s.t. s'*C*s <= delta^2

% C = U'*U; % C is identity

% solve the unconstrained problem
s = -B\a;

% check if this violate the constrain
if s'*s <= delta^2
    quad_decrease = a'*s + 0.5*s'*B*s;
    return;
end

% then reparametrize
% s1 = delta * cos t;
% s2 = delta * sin t;
%
% the model we want to minimize is
% min a1*s1 + a2*s2 + 0.5*(B11*s1^2 + B22*s2^2 + 2*B12s1*s2)
% min delta*a1*cost + delta*a2*sint + (0.5*delta^2) * (B11*cost^2 +
% B22^sint^2 + 2*B12*sintcost)
% t in [0, 2*pi)

% coarse scale searching
ts = linspace(0,2*pi,100);
fs = sub_con(a, B, delta, ts);
% find a starting point
[~, ind] = min(fs);
ct = ts(ind);
% Newton solve

for i = 1:100
    [df1, df2] = sub_con_df(a, B, delta, ct);
    nt = ct - df1/df2;
    if abs(nt-ct) < 1e-15
        break;
    else
        ct = nt;
    end
end
t = nt;
        
s = delta*[cos(t); sin(t)];
quad_decrease = a'*s + 0.5*s'*B*s;

end


function f = sub_con(a, B, delta, t)

f = delta*a(1)*cos(t) + delta*a(2)*sin(t) + ...
    (0.5*delta^2) * ( B(1,1)*cos(t).^2 + B(2,2)*sin(t).^2 + 2*B(1,2)*sin(t).*cos(t) );

end


function [df1, df2] = sub_con_df(a, B, delta, t)

df1 = -delta*a(1)*sin(t) + delta*a(2)*cos(t) + ...
    (0.5*delta^2) * ( - 2*B(1,1)*cos(t).*sin(t) + 2*B(2,2)*sin(t).*cos(t) - ...
    2*B(1,2)*sin(t).^2 + 2*B(1,2)*cos(t).^2 );

df2 = -delta*a(1)*cos(t) - delta*a(2)*sin(t) + ...
    (0.5*delta^2) * ( - 2*B(1,1)*(cos(t)^2 - sin(t)^2) + 2*B(2,2)*(cos(t)^2 - sin(t)^2) - ...
    4*B(1,2)*cos(t)*sin(t) - 4*B(1,2)*cos(t)*sin(t) );

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [jump, quad_decrease] = cauchy(a, b, d)

%d = -(delta/curr.n_grad)*curr.grad;
% p = alpha*d, alpha in [0, 1]
% the local model is
% g'*d*alpha = 0.5*alpha^2*d'*B*d
%
% min a*alpha^2 + b*alpha, alpha in [0, 1]
%
%b = curr.grad'*d;
%a = 0.5*d'*opt_def.hessmult(curr.hessinfo, d);

if a > 0
    alpha = -b/(2*a);
    if alpha < 1 && alpha > 0
        jump = d*alpha;
        quad_decrease = a*alpha^2 + b*alpha;
    elseif (a+b) < 0
        jump = d;
        quad_decrease = a+b;
    else
        jump = 0;
        quad_decrease = 0;
        disp('Cauchy solve failed: not move')
    end
else
    if (a+b) < 0
        jump = d;
        quad_decrease = a+b;
    else
        jump = 0;
        quad_decrease = 0;
        disp('Cauchy solve failed: not move')
    end
end

end
