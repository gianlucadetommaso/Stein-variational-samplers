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