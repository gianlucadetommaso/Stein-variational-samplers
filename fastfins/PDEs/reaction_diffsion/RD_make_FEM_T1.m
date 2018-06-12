function FEM = RD_make_FEM_T1(FEM, mesh)
%HEAT_MAKE_FEM_T1
%
% setup the transient heat problem with Neumann b.c.
%
% Tiangang Cui, 09/May/2014

% tol = 1e-10;

FEM     = RD_make_FEM(FEM, mesh); % setup the basic structure

%%%%%%%%% The source term, can be user defined %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% put Gaussian sink in the middle, source at 4 corners
f = 1*exp(-0.5*sum((mesh.node-[0.2;0.2]*ones(1,mesh.N_node)).^2)/0.02^2)'/(2*pi*0.02^2) + ...
    1*exp(-0.5*sum((mesh.node-[0.8;0.2]*ones(1,mesh.N_node)).^2)/0.02^2)'/(2*pi*0.02^2) + ...
    1*exp(-0.5*sum((mesh.node-[0.8;0.8]*ones(1,mesh.N_node)).^2)/0.02^2)'/(2*pi*0.02^2) + ...
    1*exp(-0.5*sum((mesh.node-[0.2;0.8]*ones(1,mesh.N_node)).^2)/0.02^2)'/(2*pi*0.02^2) - ...
    4*exp(-0.5*sum((mesh.node-[0.5;0.5]*ones(1,mesh.N_node)).^2)/0.02^2)'/(2*pi*0.02^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FEM.fs = FEM.M*f;
