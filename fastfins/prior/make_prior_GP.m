function cov = make_prior_GP(mesh, scale, power, sigma)
%MAKE_PRIOR_GP
% makes the Gaussian prior distribution for a given mesh and a correlation length
%
%%%%%%%%%%%%%%%%%%%% input: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% mesh:   
% scale:        the tensor for scaling correlation length
% power:        the power term in the kernel
% sigma:        the standard deviation
%
% Tiangang Cui, 12/May/2012

n           = size(mesh.node,2);
C           = zeros(n);
R           = chol(scale);      % tmp = inv(chol(scale)');

for i   = 1:n
    x       = mesh.node(:,i);
    d       = (mesh.node - repmat(x,1,n));
    C(:,i)  = sum((R'\d).^2);
end

if power == 1
    C       = (exp(-0.5*C) + 1e-10*eye(n));
else
    C       = exp(-0.5*C.^power);
end

cov.C       = C*sigma^2;
cov.RC      = chol(cov.C);
cov.Q       = inv(cov.C);
cov.type    = 'GP';

%{
[V D] = eigs(C,300);

d = diag(D);

ind = d>1e-12;

Cv = V(:,ind);
Ce = d(ind);
%}

end

% -0.5*(|x-y|/s)^p*scale

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%