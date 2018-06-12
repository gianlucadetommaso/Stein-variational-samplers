function map = opt(opt_def)
% MAP_SOLVER     computes the MAP estimate
%
% Tiangang Cui, 15/mar/2013

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% process the inputs
opt_def.np = length(opt_def.init);
% termination conditions
% max iter
if ~isfield(opt_def, 'max_iter')
    opt_def.max_iter = 100;
end
% KKT
if ~isfield(opt_def, 'first_KKT_tol')
    opt_def.first_KKT_tol = 1E-5;
end
if ~isfield(opt_def, 'jump_size_tol')
    opt_def.jump_size_tol = 1E-10;
end
if ~isfield(opt_def, 'fval_tol')
    opt_def.fval_tol = 1E-10;
end

% iteration solver
if ~isfield(opt_def, 'search')
    opt_def.search = 'TR';
end
if ~isfield(opt_def, 'solver')
    opt_def.solver = 'Newton';
end
switch opt_def.search
    case{'Line'}
        if ~isfield(opt_def, 'line_max_feval')
            opt_def.line_max_feval = 50;
        end
        if ~isfield(opt_def, 'line_ftol')
            opt_def.line_ftol = 1E-4;
        end
        if ~isfield(opt_def, 'line_gtol')
            opt_def.line_gtol = 0.99;
        end
        if ~isfield(opt_def, 'line_step')
            opt_def.line_step = 0.0001;
        end
    case {'TR'}
        if ~isfield(opt_def, 'TR_radius')
            opt_def.TR_radius = 100;
        end
end

if strcmp(opt_def.solver, 'Newton')
    % CG options
    if ~isfield(opt_def, 'CG_restart')
        opt_def.CG_restart = 50;
    end
    if ~isfield(opt_def, 'CG_forcing_tol')
        opt_def.CG_forcing_tol = 0.5;
    end
    if ~isfield(opt_def, 'CG_max_iter')
        opt_def.CG_max_iter = 100;
    end
    if ~isfield(opt_def, 'CG_zero_tol')
        opt_def.CG_zero_tol = 1E-6;
    end
end

% BFGS options
if strcmp(opt_def.solver, 'BFGS')
    if ~isfield(opt_def, 'bfgs_max_num')
        opt_def.bfgs_max_num = 100;
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start solve

fprintf('\n\nIteration \t Obj Value \t\t 1st KKT \t\t CG \t\n');


switch opt_def.search
    case {'Line'}
        map = line_search(opt_def);
    case {'TR'}
        map = trust_region(opt_def);
end


end
