function [mesh, mesh_dual] = rectmesh2d(gx,gy)
% RECTMESH2D   mesh generator
%
% mesh mesh_dual] = RECTMESH2D(gx,gy)
%
%%%%%%%%%%%%%%%%%%%% input: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gx, gy:       grid points on x and y axis 
%
%%%%%%%%%%%%%%%%%%%% output: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mesh:         mesh structure
% mesh_dual:    the dual mesh defined by the centers of the mesh
%
% Tiangang Cui, 03/May/2012
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = length(gx);
n = length(gy);

M_side = m-1;         % number of elements per side
N_side = n-1;         % number of elements per side

N_el = M_side*N_side;       % number of elements
N_node = m*n; % number of nodes

[xx,yy] = meshgrid(gx,gy);  % Generate mesh of node coordinates.

% Now number the nodes (from top left downwards)
coord = [xx(:)' ; yy(:)'];

% Generate local-to-global node number mapping
% Local node numbering is:
%
%   4---3
%   |   |
%   1---2
n_ref = 4;             % number of nodes on reference element
node_map = zeros(n_ref,N_el);

blnode = 1; %assemble the node map from the bottom left node
for i = 1:N_el
    node_map(:,i) = blnode + [0; N_side+1; N_side+2; 1];
    blnode = blnode + 1;
    if rem(blnode,N_side+1) == 0 % reach the top of the column (y)
        blnode = blnode+1; % skip to next column
    end
end


% Set up array of the boundary nodes
N_bnd_f = 2*(M_side + N_side);    % number of boundary faces
lb = [1:N_side; 2:N_side+1];    % left
rb = lb + (N_side+1)*M_side;    % right
bb = ([1:M_side; 2:M_side+1]-1)*(N_side+1)+1;       % bottom
tb = [1:M_side; 2:M_side+1]*(N_side+1);             % top
node_map_bnd = [fliplr(flipud(lb)), bb, rb, fliplr(flipud(tb))];

% mesh structures
mesh.Nnode = N_node;
mesh.Nel = N_el;
mesh.Nbndf = N_bnd_f;

mesh.node = coord;
mesh.node_map = node_map;
mesh.node_map_bnd = node_map_bnd;

mesh.rs = sqrt(sum(mesh.node.^2));

mesh.Mside = M_side;         % number of elements per side
mesh.Nside = N_side;         % number of elements per side

% dual mesh
mesh.centers = (mesh.node(:,mesh.node_map(1,:))+mesh.node(:,mesh.node_map(2,:))+mesh.node(:,mesh.node_map(3,:))+mesh.node(:,mesh.node_map(4,:)))/4;
mesh_dual = dual_mesh(mesh);

mesh.gx = gx;
mesh.gy = gy;

mesh_dual.gx = 0.5*(gx(1:end-1)+gx(2:end));
mesh_dual.gy = 0.5*(gy(1:end-1)+gy(2:end));

mesh        = make_local(mesh);
mesh_dual   = make_local(mesh_dual);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function mesh = dual_mesh(mesh_f)

mesh.node = mesh_f.centers;
m = mesh_f.Mside;
n = mesh_f.Nside;

mesh.node_map = zeros(4,(m-1)*(n-1));

for i = 1:m-1
    for j = 1:n-1
        k = i*(n+1)+j+1;
        % given a node, search for the elementts attached to it
        i1 = find(mesh_f.node_map(3,:) == k); % bottom left, v3 
        i2 = find(mesh_f.node_map(4,:) == k); % bottom right, v4
        i3 = find(mesh_f.node_map(1,:) == k); % top right, v1
        i4 = find(mesh_f.node_map(2,:) == k); % top left, v2
        mesh.node_map(:,(i-1)*(n-1)+j) = [i1 i2 i3 i4];
    end
end

mesh.Nnode = m*n;
mesh.Nel = (m-1)*(n-1);
mesh.Nbndf = 0;

mesh.Mside = m-1;
mesh.Nside = n-1;

%mesh.node_map = delaunay(centers(1,:),centers(2,:));

mesh.centers = (mesh.node(:,mesh.node_map(1,:))+mesh.node(:,mesh.node_map(2,:))+mesh.node(:,mesh.node_map(3,:))+mesh.node(:,mesh.node_map(4,:)))/4;
mesh.node_map_bnd = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function mesh = make_local(mesh)
% MAKE_LOCAL    generates the local reference bilinear element
%
% mesh = MAKE_LOCAL(mesh)
%
%%%%%%%%%%%%%%%%%%%% inputs: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mesh:     the FEM mesh
%%%%%%%%%%%%%%%%%%%% outputs: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mesh:     thi subroutine modifies the mesh object
%
% Tiangang Cui, 03/May/2012

mesh.w1 = [1 -1 -1 1]'/2;
mesh.w2 = [-1 -1 1 1]'/2;
mesh.w3 = [1 -1 1 -1]'/2*sqrt(2/3);

mesh.locstiff = [4  -1 -2 -1;
                 -1  4 -1 -2;
                 -2 -1  4 -1;
                 -1 -2 -1  4]/6;
        
mesh.locmass = [4, 2, 1, 2;
                2, 4, 2, 1;
                1, 2, 4, 2;
                2, 1, 2, 4]/36;
            
mesh.locmass_bnd = [2, 1;
                    1, 2]/6;
                
mesh.locadvx = [ -2, -2, -1, -1;
                  2,  2,  1,  1;
                  1,  1,  2,  2;
                 -1, -1, -2, -2]/12;
                
mesh.locadvy = [ -2, -1, -1, -2;
                 -1, -2, -2, -1;
                  1,  2,  2,  1;
                  2,  1,  1,  2]/12;
                
mesh.v1 = [-1 1 -1 1]'/2/6;
mesh.v2 = [0 -1 0 1]'/sqrt(2)/sqrt(12);
mesh.v3 = [1 0 -1 0]'/sqrt(2)/sqrt(12);
mesh.v4 = [1 1 1 1]'/2/2;

mesh.loc.xx = [ 2 -2 -1  1;
               -2  2  1 -1;
               -1  1  2 -2;
                1 -1 -2  2]/6;

mesh.loc.yy = [ 2  1 -1 -2;
                1  2 -2 -1;
               -1 -2  2  1;
               -2 -1  1  2]/6;

mesh.loc.xy = [ 1  0 -1  0;
                0 -1  0  1;
               -1  0  1  0;
                0  1  0 -1]/2;

%mesh.locmass*mesh.v1/sqrt(1/36) - mesh.v1*sqrt(1/36)
%mesh.locmass*mesh.v2/sqrt(1/12) - mesh.v2*sqrt(1/12)
%mesh.locmass*mesh.v3/sqrt(1/12) - mesh.v3*sqrt(1/12)
%mesh.locmass*mesh.v4/sqrt(1/4) - mesh.v4*sqrt(1/4)
%[mesh.v1 mesh.v2 mesh.v3 mesh.v4]'*[mesh.v1 mesh.v2 mesh.v3 mesh.v4] - diag([1/36 1/12 1/12 1/4])
end