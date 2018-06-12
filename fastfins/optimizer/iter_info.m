function info = iter_info(opt_def, f0, curr, next, i, iter)
% ITER_INFO  determines the terminition condition
%
% Tiangang Cui, 16/Mar/2013

info = 0;

tmp = norm(next.grad)/opt_def.np;

%if isempty(a)
    jump = norm(next.x - curr.x)/opt_def.np;
%else
%    jump = a;
%end

% display iteration info
fprintf('%4i \t\t %10.5E \t\t %10.5E \t\t %4i\n', [i, next.f, tmp, iter]);

if tmp < opt_def.first_KKT_tol
    info = 1;
    disp('First KKT condition satisfied, exit');
    return;
end

if jump < opt_def.jump_size_tol
    info = 2;
    disp('Jump size reach minumum threshold, exit')
    return
end

if abs(curr.f - next.f)/abs(f0) < opt_def.fval_tol
    info = 3;
    disp('Objective function does not have suffcient decrease, exit');
    return
end

if i > opt_def.max_iter
    info = 4;
    disp('Maximum number of iteration reached, no optimal solution find');
    return;
end

end