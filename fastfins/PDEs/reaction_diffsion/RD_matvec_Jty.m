function Jty = RD_matvec_Jty(FEM, HI, dy)
%HEAT_MATVEC_JTY
%
% Adjoint solve
%
% Tiangang Cui, 09/May/2014 

RESTART = 3;
TOL     = 1E-6;
MAXIT   = 50;

N       = size(dy, 2)/FEM.Ndatasets;
Jty     = zeros(FEM.DoF, N);

for i   = 1:N
    ind = (i-1)*FEM.Ndatasets + (1:FEM.Ndatasets);
    dU  = FEM.C'*(dy(:,ind)*HI.T');
    
    for j = (HI.Nend+1):-1:2
        
        dt              = HI.dts(j-1);
    
        if  j   == (HI.Nend+1)
            g           = - dU(:,j);
        else
            if FEM.ML_flag
                g       = Jty(:,i) - dU(:,j);
            else
                g       = FEM.M*Jty(:,i) - dU(:,j);
            end
        end
        
        if FEM.fast_adj_flag
            if FEM.GMRES_flag
                J       = dGds(FEM, HI.G(:,j), dt)';
                Jty(:,i)= gmres(J, g, RESTART, TOL, MAXIT, HI.Us{j-1}', HI.Ls{j-1}');
            else
                Jty(:,i)= HI.Ls{j-1}'\(HI.Us{j-1}'\g);
            end
        else
            J           = dGds(FEM, HI.G(:,j), dt)';
            if FEM.GMRES_flag
                [L, U]  = make_precond(J);
                Jty(:,i)= gmres(J, g, RESTART, TOL, MAXIT, L, U);
            else
                Jty(:,i)= J\g;
            end
        end
    end
    
    if FEM.ML_flag
        Jty(:,i)        = dU(:,1) - Jty(:,i);
    else
        Jty(:,i)        = dU(:,1) - FEM.M*Jty(:,i);
    end
    
end

end


