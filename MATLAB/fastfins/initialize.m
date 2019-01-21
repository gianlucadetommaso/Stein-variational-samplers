function [model, prior] = initialize(model_def, prior_def, obs_def, output)
% 
% Initialization script
% Tiangang Cui, 21/Jan/2014
%
% This version only focuses on using the single type of spatially distibuted
% parameters.
%
%%%%%%%%%

%%%%%%%%%%%%%%%%%%% start of mesh generation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 gx                 = linspace(0,model_def.xyratio,model_def.xyratio*model_def.mesh_size+1);
 gy                 = linspace(0,1,model_def.mesh_size+1);
[mesh, mesh_dual]   = rectmesh2d(gx,gy); % mesh maker
 mesh               = make_local(mesh); % assign bilinear element
 mesh_dual          = make_local(mesh_dual); 

%%%%%%%%%%%%%%%%%%% initialize parameter and prior %%%%%%%%%%%%%%%%%%%%%%%%

prior               = init_prior_dist(prior_def, mesh_dual);
param.true_x        = init_test_image(mesh, prior_def, output, prior);

%%%%%%%%%%%%%%%%%%% initialize forward problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch model_def.problem
    case{'Laplace'}
        [FEM, obs, ref_soln] = init_laplace(model_def, obs_def, output, mesh, param);
    case{'EIT'}
        [FEM, obs, ref_soln] = init_EIT(model_def, obs_def, output, mesh, param);
    case{'Heat_single'}
        [FEM, obs, ref_soln] = init_heat(model_def, time_def, obs_def, output, mesh, param);
end

% setup the problem
model.mesh      = mesh;
model.mesh_dual = mesh_dual;
model.obs       = obs;
model.FEM       = FEM;
model.ref_soln  = ref_soln;

% setup relevant functions
model.forward       = @(x)      forward_solve        (mode.FEM, x);
model.adjoint       = @(s, m)   adjoint_grad         (model.mesh, model.obs, model.FEM, s, m);
model.matvec_GNH    = @(H, dx)  adjoint_hessmult_NP  (model.mesh, model.obs, model.FEM, H, dx);
model.matvec_Jty    = @(H, dy)  adjoint_jacmult_left (model.mesh, model.obs, model.FEM, H, dy);
model.matvec_Jx     = @(H, dx)  adjoint_jacmult_right(model.mesh, model.obs, model.FEM, H, dx);
model.explicit_J    = @(H)      adjoint_jacobian     (model.mesh, model.obs, model.FEM, H);

model.SVD.Jm        = model.obs.N_elect;
model.SVD.Jn        = prior.DoF; 
model.SVD.num_rand  = model.obs.N_obs + 5;

%%%%%%%%%%%%%%%%%%% end of forward problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end