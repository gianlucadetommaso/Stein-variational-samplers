function [ymap, xmap] = MAP(mesh, mesh_dual, obs, prior, param, forward)
% MAP   Runs optimization algorithms to get the MAP estitimate
%
% [zmap, xmap] = MAP(mesh, mesh_dual, obs, prior, param, forward)
%
% Tiangang Cui, 04/May/2012

%{
if strcmp(param.type,'Affine_log')
    param.type = 'Transformed_orth';
    param.basis = param.basis_GP;
    param.DoF = param.DoF_GP;
    forward.solver = 'FOM';
    forward.NoP = param.NP;
end
%}

test_MAP = true;

y_init = randn(param.NP,1);

out = pre_process(prior, param, y_init, false);
soln = forward_solve(forward, out.x);
d_init = soln.d-obs.d;

tic
ymap = get_map(mesh, obs, prior, param, forward, y_init);
toc

out = pre_process(prior, param, ymap, false);
xmap = out.x;
soln = forward_solve(forward, xmap);
d_end = soln.d-obs.d;

% plot
figure
subplot(3,1,1);plot(d_init');title('init error')
subplot(3,1,2);plot(d_end');title('final error')
subplot(3,1,3);plot(obs.d');title('data')

figure('position',[100 100 700 300]);
subplot(1,2,1)
switch param.type
    %case{'Affine_log'}
    %    xmap = param_z2x_simple( param, zmap );
    %    meshc(mesh_dual.gx,mesh_dual.gy,reshape(param.basis*xmap,mesh_dual.N_side+1,mesh_dual.N_side+1))
    case{'Affine','Affine_log'}
        meshc(mesh_dual.gx,mesh_dual.gy,reshape(param.basis*xmap,mesh_dual.M_side+1,mesh_dual.N_side+1))
    otherwise
        meshc(mesh_dual.gx,mesh_dual.gy,reshape(xmap,mesh_dual.M_side+1,mesh_dual.N_side+1))
end
if exist('prior.beta','var')
    title(['MAP - \beta = ' num2str(prior.beta)])
else
    title('MAP')
end
%%%%%%%%%%%
subplot(1,2,2)
switch param.type
    case{'Affine','Affine_log'}
        imagesc(mesh_dual.gx,mesh_dual.gy,reshape(param.basis*xmap,mesh_dual.M_side+1,mesh_dual.N_side+1));
    otherwise
        imagesc(mesh_dual.gx,mesh_dual.gy,reshape(xmap,mesh_dual.M_side+1,mesh_dual.N_side+1));
end
set(gca,'ydir','normal');
xlabel('x')
ylabel('y')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if test_MAP
    plot_minus_log_post(mesh, obs,prior,param,forward,ymap,10);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
% low rank Laplace at the MAP
post = prior;

if low_rank_n > 0
[V d] = minus_llkd_hess(mesh, obs, prior, param, Model, zmap, low_rank_n);

% form the fake posterior
if prior.on_off
    post.mean = zmap;
    post.P = prior.beta*prior.P + V*diag(d)*V';
    post.type = 'MRF'; % use the precision
    post.RP = chol(post.P);
    post.basis = V;
    post.eigen = d;
end
%}

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_minus_log_post(mesh, obs,prior,param,forward,x,n)
% PLOT_MINUS_LOG_POST  plot the shape of the objective function around a give
%                      point x
%
% PLOT_MINUS_LOG_POST(obs,prior,param,Model,z)
%
%%%%%%%%%%%%%%%%%%%% input: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% obs:      the observation structure
%
% prior:    the prior structure
%
% param:    the parameter structure
%
% Model:    the FEM/ROM structure
%
% x:        input parameter
%
% Tiangang Cui, 03/June/2012


figure
for j = 1:n
    % draw a rando mdirectoin at Hessian
    dx = randn(size(x));
    % normalize
    ddp = dx/norm(dx);
    
    dr = linspace(-2,2,20);
    obj = zeros(size(dr));
    
    for i = 1:length(dr)
        z = x+ddp*dr(i);
        obj(i) = minus_log_post_mcmc(mesh, obs, prior, param, forward, z);
    end
    
    plot(dr,obj);
    hold on
end
end
