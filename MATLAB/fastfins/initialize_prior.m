function prior = initialize_prior(model_def, prior_def)
%
% Initialization script for the Heat conduction problem
% Tiangang Cui, 21/Jan/2014
%
% This version only focuses on using the single type of spatially distibuted
% parameters.
%
%%%%%%%%%

gx      = linspace(0,model_def.xyratio,model_def.xyratio*model_def.mesh_size+1);
gy      = linspace(0,1,model_def.mesh_size+1);
mesh    = rectmesh2d(gx,gy); % mesh maker
mesh    = make_local(mesh); % assign bilinear element
prior   = init_prior_dist(prior_def, mesh_dual);
prior.mesh  = mesh;

end