function FEM = RD_make_FEM(FEM, mesh)
%HEAT_MAKE_FEM  
%
% makes the basic FEM structure 
%
% Tiangang Cui, 31/Oct/2012


FEM.W1              = sparse([],[],[],mesh.Nnode,mesh.Nel,4*mesh.Nel);
FEM.W2              = sparse([],[],[],mesh.Nnode,mesh.Nel,4*mesh.Nel);
FEM.W3              = sparse([],[],[],mesh.Nnode,mesh.Nel,4*mesh.Nel);

FEM.V1              = sparse([],[],[],mesh.Nnode,mesh.Nel,4*mesh.Nel);
FEM.V2              = sparse([],[],[],mesh.Nnode,mesh.Nel,4*mesh.Nel);
FEM.V3              = sparse([],[],[],mesh.Nnode,mesh.Nel,4*mesh.Nel);
FEM.V4              = sparse([],[],[],mesh.Nnode,mesh.Nel,4*mesh.Nel);

FEM.M               = sparse([],[],[],mesh.Nnode,mesh.Nnode,10*mesh.Nel);
FEM.Mb              = sparse([],[],[], mesh.Nnode, mesh.Nnode, 4*mesh.Nbndf);

for               i = 1:mesh.Nbndf
    ind             = mesh.node_map_bnd(:,i);
    dx              = mesh.node(:,ind(2)) - mesh.node(:,ind(1));
    FEM.Mb  (ind,ind)   = FEM.Mb(ind,ind) + norm(dx)*mesh.locmass_bnd;
end

for               i = 1:mesh.Nel
    ind             = mesh.node_map(:,i);
    dx              = mesh.node(:,ind(3)) - mesh.node(:,ind(1));
    detJ            = prod(abs(dx));    % iJ = diag(1./dx);
        
    FEM.W1(ind,i)   = mesh.w1;
    FEM.W2(ind,i)   = mesh.w2;
    FEM.W3(ind,i)   = mesh.w3;
    
    FEM.M(ind,ind)  = FEM.M(ind,ind) + mesh.locmass*detJ;  
    
    FEM.V1(ind,i)   = mesh.v1*sqrt(detJ);
    FEM.V2(ind,i)   = mesh.v2*sqrt(detJ);
    FEM.V3(ind,i)   = mesh.v3*sqrt(detJ);
    FEM.V4(ind,i)   = mesh.v4*sqrt(detJ);
end

FEM.const_detJ      = detJ;

%c = sparse(mesh.node_map_bnd(1,:),1,1/mesh.Nbndf,mesh.Nnode,1) + ...
%    sparse(mesh.node_map_bnd(2,:),1,1/mesh.Nbndf,mesh.Nnode,1); % projection is now c*c'
%FEM.c              = c*c'; % penalty matrix

FEM.DoF             = mesh.Nnode;

FEM.K               = FEM.W1*FEM.W1' + FEM.W2*FEM.W2' + FEM.W3*FEM.W3';
FEM.I               = speye(FEM.DoF);
FEM.ML              = spdiags(sum(FEM.M, 2), 0, FEM.DoF, FEM.DoF);
FEM.iML             = spdiags(1./sum(FEM.M, 2), 0, FEM.DoF, FEM.DoF);
FEM.iMLK            = FEM.iML*FEM.K;


FEM.mesh            = mesh;

end


