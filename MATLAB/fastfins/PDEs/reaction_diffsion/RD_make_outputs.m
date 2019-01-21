function FEM = RD_make_outputs(mesh, obs_def)
%HEAT_MAKE_OUTPUTS
%
% generates measurements points and indices w.r.t. the FEM mesh
%
% Tiangang Cui, 03/May/2014

tol             = 1E-10;

switch  obs_def.type
    case {1}
        [xx,yy] = meshgrid(obs_def.locs, obs_def.locs);
        locs    = [xx(:) yy(:)];
        %ind = rand(size(temp,1),1)<1.1;
        %locs = temp(ind,:);
        
        FEM.sensors         = zeros(size(locs,1),1);
        
        for i   = 1:size(locs,1)
            ind = find(sum(abs(mesh.node - locs(i,:)'*ones(1,mesh.N_node)))<tol);
            FEM.sensors(i)  = ind;
        end
        
    case {2}
        ind     = mesh.node(1,:)>(obs_def.locs(1)+tol) & mesh.node(1,:)<(obs_def.locs(2)-tol) & ...
                  mesh.node(2,:)>(obs_def.locs(3)+tol) & mesh.node(2,:)<(obs_def.locs(4)-tol);
        FEM.sensors         = find(ind);
        
    case {3}
        ind     = mesh.node(1,:)>tol & mesh.node(1,:)<(1-tol) & ...
                  mesh.node(2,:)>tol & mesh.node(2,:)<(1-tol);
        FEM.sensors         = find(ind);
    case {4}
        FEM.sensors = zeros(size(obs_def.locs,1),1);
        for i = 1:size(obs_def.locs,1)
            ind     = find(sum(abs(mesh.node - obs_def.locs(i,:)'*ones(1,mesh.Nnode)))<tol);
            FEM.sensors(i)  = ind;
        end
    case {5}
        N           = size(obs_def.locs,1);
        FEM.C       = [];
        for k = 1:N
            ind     = mesh.centers(1,:)>(obs_def.locs(k,1)+tol) & mesh.centers(1,:)<(obs_def.locs(k,2)-tol) & ...
                      mesh.centers(2,:)>(obs_def.locs(k,3)+tol) & mesh.centers(2,:)<(obs_def.locs(k,4)-tol);
            tmp     = 1:mesh.Nel;
            elems   = tmp(ind);
            C       = sparse([],[],[],1,mesh.Nnode,ceil((sqrt(sum(ind))+1)^2));
            for i = 1:sum(ind)
                nodes   = mesh.node_map(:,elems(i));
                dx      = mesh.node(:,nodes(3)) - mesh.node(:,nodes(1));
                detJ    = prod(abs(dx)); % iJ = diag(1./dx);
                locs    = 0.25*detJ*ones(1,4);
                C(1,nodes)  = C(1,nodes) + locs;
            end
            FEM.C   = [FEM.C; C];
        end
        
end

if  obs_def.type == 5
    FEM.Nsensors        = size(FEM.C, 1);
else
    FEM.Nsensors        = length(FEM.sensors);
    FEM.C               = sparse([],[],[], FEM.Nsensors, mesh.Nnode, FEM.Nsensors); % make the observation matrix
    FEM.C(:,FEM.sensors)= speye(FEM.Nsensors);
end
FEM.Tstart              = obs_def.Tstart;
FEM.Tfinal              = obs_def.Tfinal;
FEM.Ndatasets           = obs_def.Ntime;

end