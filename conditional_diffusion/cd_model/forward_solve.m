function soln = forward_solve(model, w)
%FORWARD
%
% Solves the SDE for a given realization of the random forcing w
% run ``mex f_kody.c'' to compile the code for the SDE 
% Tiangang Cui, 20/Nov/2013

soln.G      = euler_solve(model.N, model.d, model.dt, w(:));
soln.d      = soln.G((model.k+1):model.k:end);
soln.udu    = linear_sparse(model, soln.G);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function udu = linear_sparse(model, u)
%LINEAR_SPARSE
%
% linearized forward model, sparse 
%
% Tiangang Cui, 20/Nov/2013

N    = model.N+1;
dfdu = df_kody_du(model.d, u);
off  = -1-dfdu(1:(end-1))*model.dt;
udu  = sparse([1:N 2:N],[1:N 1:(N-1)], [ones(1,N), off(:)'], N, N);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dfdu = df_kody_du(d, u)
%DF_KODY_DU
%
% non-linear drifting of the SDE
%
% Tiangang Cui, 20/Nov/2013
% Modified from Kody's orginal code

tmp  = u.^2;
dfdu = d*( (1 - 3*tmp)./(1 + tmp ) - 2*tmp.*(1 - tmp)./(1 + tmp).^2 );

end

