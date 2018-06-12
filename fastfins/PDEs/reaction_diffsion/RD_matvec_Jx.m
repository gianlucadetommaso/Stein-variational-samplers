function Jx = RD_matvec_Jx(FEM, HI, dx)
%HEAT_MATVEC_JX
%
% Adjoint solve
%
% Tiangang Cui, 09/May/2014 

RESTART = 3;
TOL     = 1E-6;
MAXIT   = 50;

N       = size(dx, 2);
ds      = zeros(FEM.DoF,      HI.Nend+1);
Jx      = zeros(FEM.Nsensors, FEM.Ndatasets*N);

for i   = 1:N
    ind = (i-1)*FEM.Ndatasets + (1:FEM.Ndatasets);
    
    ds(:,1)                 = dx(:,i);
    for j = 1:HI.Nend
        dt                  = HI.dts(j);
        
        if FEM.ML_flag
            g = ds(:,j);
        else
            g = FEM.M*ds(:,j);
        end
        
        if FEM.fast_adj_flag
            if FEM.GMRES_flag
                J           = dGds(FEM, HI(:,i+1), dt);
                ds(:,j+1)   = gmres(J, g, RESTART, TOL, MAXIT, HI.Ls{j}, HI.Us{j});
            else
                ds(:,j+1)   = HI.Us{j}\(HI.Ls{j}\g);
            end
        else
            J               = dGds(FEM, HI(:,i+1), dt);
            if FEM.GMRES_flag
                [L, U]      = make_precond(J);
                ds(:,j+1)   = gmres(J, g, RESTART, TOL, MAXIT, L, U);
            else
                ds(:,j+1)   = J\g;
            end
        end
    end
    
    Jx(:,ind)               = (FEM.C*ds)*HI.T;
end


end