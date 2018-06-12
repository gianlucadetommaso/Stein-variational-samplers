function soln = RD_solve(FEM, s_init)
%HEAT_SOLVE  
%
% solves the transient heat equation by Cholesky and implicit Euler
%
% Tiangang Cui, 09/May/2014

RESTART             = 3;
TOL                 = 1E-6;
MAXIT               = 50;
NR_tol              = 1E-8;

soln.G              = zeros(FEM.DoF, FEM.Nmaxsteps+1);
soln.G(:,1)         = s_init;

dt                  = FEM.dt_init;
time                = 0;
soln.Nend           = 0;
soln.dts            = zeros(FEM.Nmaxsteps, 1);

if FEM.fast_adj_flag
    soln.Ls         = cell(FEM.Nmaxsteps, 1);
    soln.Us         = cell(FEM.Nmaxsteps, 1);
end

for i = 1:FEM.Nmaxsteps
    % Newton iteration
    NR_iter         = 0;
    s               = soln.G(:,i);
    
    while 1
        G           = eval_G(FEM, soln.G(:,i), s, dt);
        J           = dGds(FEM, s, dt);
        if FEM.GMRES_flag
            [L, U]  = make_precond(J);
            ds      = gmres(J, -G, RESTART, TOL, MAXIT, L, U);
        else
            [L, U]  = lu(J);
            ds      = -U\(L\G);
        end
 
        s       = s+ds;
        NR_iter = NR_iter + 1;
        
        if ( ds'*(FEM.ML*ds) )  < NR_tol^2
            break;
        end
    end
    
    % NR_iter
    
    if FEM.fast_adj_flag
        soln.Ls{i}  = L;
        soln.Us{i}  = U;
    end
    
    soln.G(:,i+1)   = s; 
    time            = time + dt;
    soln.dts(i)     = dt;
    
    if time >= FEM.Tfinal
        soln.Nend   = i;
        break;
    end
    
    if NR_iter <= 3
        dt          = min(dt * FEM.dt_multiplier, FEM.dt_max);
    end
end


% process observations 
obs_Tsteps          = linspace(FEM.Tstart, FEM.Tfinal, FEM.Ndatasets);
soln.T              = sparse([],[],[], soln.Nend+1, FEM.Ndatasets, FEM.Ndatasets*2);
t_start             = 0;
for i = 1:soln.Nend
    t_end           = t_start + soln.dts(i);
    t_ind           = find(obs_Tsteps>t_start & obs_Tsteps<=t_end);
    if sum(t_ind) > 0
        temp1               = (t_end  -  obs_Tsteps(t_ind))/soln.dts(i); % weighting at t_start
        temp2               = (obs_Tsteps(t_ind) - t_start)/soln.dts(i); % weighting at t_end
        soln.T(i,  t_ind)   = temp1;
        soln.T(i+1,t_ind)   = temp2;
    end
    t_start         = t_end; % increment the time
end

soln.G              = soln.G(:,1:soln.Nend+1);
soln.d              = (FEM.C*soln.G)*soln.T;

end

