function prior = make_prior(Tend, Nstep)
%MAKE_PRIOR
%
% build the forward difference matrix, and the covariance matrix of the
% Brownian motion part.
%
% Tiangang Cui, 18/May/2014

dt              = Tend/Nstep;
ts              = linspace(0,1,Nstep+1)*Tend;    % The time grid
[T,S]           = meshgrid(ts,ts);               % mesh grid
C_tmp           = min(T,S);                      % the covariance of the brownian motion
% using Schur Complemtent to get conditional
% C      = C_tmp(2:end,2:end) - C_tmp(2:end,1)*C_tmp(1,1)^(-1)*C_tmp(1,2:end); 
prior.cov.C     = C_tmp(2:end,2:end);

D               = spdiags([-ones(Nstep,1), ones(Nstep,1)], [0 1], Nstep, Nstep+1)/sqrt(dt); % forward difference
%D_mod  = D(:,2:end); % forward difference, modified

e               = ones(Nstep+1,1)/dt; % for generating Laplace
L2_tmp          = spdiags([e -2*e e], -1:1, Nstep+1, Nstep+1); % Laplace operator
L2_tmp(1,1)     = -1/dt;
L2_tmp(end,end) = -1/dt;
prior.cov.P     = -L2_tmp(2:end, 2:end); % boundary condition


prior.cov.L     = tril(ones(Nstep))*sqrt(dt); % cumsum operator
[V, D]          = eig(prior.cov.C);
[d,ind]         = sort(diag(D), 'descend');
prior.T         = prior.cov.L'*V(:,ind);

% w'*prior.T gives the parameters on the KL modes

figure
semilogy(d);
title('prior eigenvalues')

prior.NP        = Nstep;
prior.DoF       = Nstep;
prior.type      = 'Dist';

prior.mean_u    = zeros(prior.DoF, 1);
prior.func.type = 'None';

end

% D*C_tmp*D' = I
% D'*D + L2_tmp = 0

%{
dt=10/N;
[T,S]=meshgrid(ts,ts);
e = [1:N].^0;
e = ones(1,N+1);
C = min(T,S);
C12=sqrtm(C);
[V,E]=eig(C12);

L=double(sqrt(dt)*(T<=S));

D=(diag(-e(1:end-1)',-1)+diag(e(1:end)',0));



L2=(1/dt)*(diag(-e(1:end-1)',-1)+2*diag(e(1:end)',0)+...
    diag(-e(1:end-1)',1));L2(end,end)=1/dt;

%}