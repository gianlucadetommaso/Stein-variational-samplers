function [U, s, V, HI] = rand_svd_J(model, data, x, tol, nmax)
%SVD_WHITENING_F_RAND
%
% randomized SVD for factorizing the forward model
% requires the HI by runing 
% [~,~,~, HI] = minus_log_post(model, prior, v);
%
% Tiangang Cui, 17/Jan/2014

[~,~,HI]    = minus_log_post(model, data, x);

if model.N_w >= data.N_y
    [U, s, V] = more_param(model, data, HI, tol, nmax);
else
    [U, s, V] = more_obs  (model, data, HI, tol, nmax);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U, s, V] = more_param(model, data, HI, tol, nmax)

% Let O*O = Gamma_{obs}^{-0.5}, L*L' = \Gamma_{pr}
% J  = O *F*diag(dxdu) *L
% Jt = L'*diag(dxdu)*F'*O'

 Nrand  = min(nmax, data.N_y+5);
 
% apply J' to data space perturbation
 Jty    = zeros(model.N_w, Nrand);
 O      = randn(data.N_y, Nrand);
 for i = 1:Nrand
     Jty(:,i)   = matvec_Jty(model, HI, O(:,i)' );
 end

% apply J to param space perturbation
[Q, ~]  = qr(Jty,0);
 Jx     = zeros(data.N_y, Nrand);
for i = 1:Nrand
    Jx(:,i)     = matvec_Jx(model, HI, Q(:,i))*HI.gamma;
end

% Jx    = J*Q = A * T * B'
% J     = J*Q*Q'
% J     = A * T * B'*Q'
 
[A, T, B]  = svd(Jx); % final svd
 
 dT         = diag(T);
 ind        = find(dT>=tol); % truncate
 U          = A(:,ind);
 V          = Q*B(:,ind);
 s          = dT(ind);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U, s, V] = more_obs(model, data, HI, tol, nmax)

% Let O*O = Gamma_{obs}^{-0.5}, L*L' = \Gamma_{pr}
% J  = O *F*diag(dxdu) *L
% Jt = L'*diag(dxdu)*F'*O'

 Nrand  = min(nmax, model.N_w+5);
 
% apply J to param space perturbation
 O      = randn(model.N_w, Nrand);
 Jx     = zeros(data.N_y, Nrand);
for i = 1:Nrand
    Jx(:,i)     = matvec_Jx(model, HI, O(:,i));
end
 
% apply J' to data space perturbation
[Q, ~]  = qr(Jx,0);
 Jty    = zeros(model.N_w, Nrand);
 for i = 1:Nrand
     Jty(:,i)   = matvec_Jty(model, HI, Q(:,i)')*HI.gamma;
 end

% Jty   = J'*Q = A * T * B'
% J'    = J'*Q*Q'
% J'    = A * T * B'*Q' 
%       = V * T * U'

[A, T, B]  = svd(Jty); % final svd
 
 dT         = diag(T);
 ind        = find(dT>=tol); % truncate
 V          = A(:,ind);
 U          = Q*B(:,ind);
 s          = dT(ind);
 
end