function [Q, prec] = make_prior_MRF_old(mesh, k, sigma, cond)
% mAKE_PRIOR_RUE_REC    makes the GMRF prior distribution for a rectangular
%                       grid 
%
% P = MAKE_PRIOR_RUE_REC(mesh, mesh_ref, gamma, scale)
%
%%%%%%%%%%%%%%%%%%%% input: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mesh:     is the dual mesh 
%
% mesh_ref: the reference mesh
%
% gamma:    variable controls the eigen spectrum of the prior
%
% sigma:        the standard deviation
%
%%%%%%%%%%%%%%%%%%%% output: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% P:        the precision matrix
%
% Tiangang Cui, 12/May/2012

[K, ML] = make_matrix(mesh, cond);

%{
xmin = min(mesh.node(1,:));
xmax = max(mesh.node(1,:));
ymin = min(mesh.node(2,:));
ymax = max(mesh.node(2,:));

tol = 1e-6;

ind_lb = find(abs(mesh.node(1,:)-xmin) < tol);
ind_rb = find(abs(mesh.node(1,:)-xmax) < tol);
ind_bb = find(abs(mesh.node(2,:)-ymin) < tol);
ind_tb = find(abs(mesh.node(1,:)-ymax) < tol);

ind = [ind_lb,ind_rb,ind_bb,ind_tb];

c = sparse(ind,1,1/length(ind),mesh.N_node,1);
c = c*c'*mesh.N_node;
%}

sigma2 = gamma(1)/(gamma(2)*(k^2)*(4*pi));
tmp    = sigma2/sigma^2;

spML    = spdiags(ML,0,mesh.N_node,mesh.N_node);
ispML   = spdiags(ML.^(-1),0,mesh.N_node,mesh.N_node);
isqspML = spdiags(ML.^(-0.5),0,mesh.N_node,mesh.N_node);

%P  = (K + c) + spML*(k^2);
P  = K + spML*(k^2);
Q  = P*(ispML*tmp)*P;

prec.per = symamd(P); % this is a MatLab function
prec.R   = chol(P(prec.per,prec.per)); % RP
prec.sca = sqrt(ML(prec.per)/tmp);
prec.RP  = sqrt(tmp)*isqspML*P;

% isqspML = spdiags(ML.^(-0.5),0,mesh.N_node,mesh.N_node);
% L = P*isqspML*sqrt(tmp);
% R = sqrt(tmp)*isqspML*P;
%
% P = RP'*RP
% sqrt(Q)_right = sqrt(tmp)*isqspML*P = sqrt(tmp)*isqspML*RP'*RP
% inv(sqrt(tmp)*isqspML) = tmp^(-0.5) * inv(isqspML) = scaling
% inv(sqrt(Q)_right) y  = RP\(RP'\(scaling*y))
% inv(sqrt(Q)_right') y = scaling*(RP\(RP'\y))

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [K, ML] = make_matrix(mesh, cond)

%{
tic;
K = sparse([],[],[],mesh.N_node,mesh.N_node,4*mesh.N_el);
M = sparse([],[],[],mesh.N_node,mesh.N_node,4*mesh.N_el);

for i = 1:mesh.N_el
    ind = mesh.node_map(:,i);
    
    % assume const and identity Jacobian
    dx = mesh.node(:,ind(3)) - mesh.node(:,ind(1));
    detJ = prod(abs(dx));
    %K(ind,ind) = K(ind,ind) + mesh.locstiff*cond(i);
    K(ind,ind) = K(ind,ind) + mesh.loc.xx*cond(i,1) + mesh.loc.yy*cond(i,2) + mesh.loc.xy*cond(i,3);
    M(ind,ind) = M(ind,ind) + mesh.locmass*detJ;
end

toc
%}

%tic;
ind_i = zeros(4^2, mesh.N_el);
ind_j = zeros(4^2, mesh.N_el);
detJs = zeros(1, mesh.N_el);

for i = 1:mesh.N_el
    ind = mesh.node_map(:,i);
    
    % assume const and identity Jacobian
    dx = mesh.node(:,ind(3)) - mesh.node(:,ind(1));
    detJ = prod(abs(dx));
    
    ii = repmat(ind', 4, 1);
    jj = repmat(ind,1, 4);
    ind_i(:,i) = ii(:);
    ind_j(:,i) = jj(:);
    detJs(i)   = detJ;
end

temp_K = mesh.loc.xx(:)*cond(:,1)' + mesh.loc.yy(:)*cond(:,2)' + mesh.loc.xy(:)*cond(:,3)';
K = sparse(ind_i(:), ind_j(:), temp_K(:), mesh.N_node, mesh.N_node);

temp_M = mesh.locmass(:)*detJs;
M = sparse(ind_i(:), ind_j(:), temp_M(:), mesh.N_node, mesh.N_node);

%toc;

ML = full(sum(M,2));

end
