function [U, s, V] = svd_rand_J(model, obs, prior, v, tol, nmax)
%SVD_WHITENING_F_RAND
%
% randomized SVD for factorizing the forward model
% requires the HI by runing 
% [~,~,~, HI] = minus_log_post(model, prior, v);
%
% Tiangang Cui, 17/Jan/2014

[~,~,~,~,HI]= minus_log_post(model, obs, prior, v);

if prior.DoF >= obs.Ndata
    [U, s, V] = more_param(model, obs, prior, HI, tol, nmax);
else
    [U, s, V] = more_obs  (model, obs, prior, HI, tol, nmax);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U, s, V] = more_param(model, obs, prior, HI, tol, nmax)

% Let O*O = Gamma_{obs}^{-0.5}, L*L' = \Gamma_{pr}
% J  = O *F*diag(dxdu) *L
% Jt = L'*diag(dxdu)*F'*O'

 Nrand      = min(nmax, obs.Ndata+5);
 
% apply J' to data space perturbation
 Jxty       = zeros(prior.DoF, Nrand);
 O          = randn(obs.Ndata, Nrand);
 for i = 1:Nrand
     Jxty(:,i)  = matvec_Jty(model, HI, O(:,i)./obs.std);
 end

% apply J to param space perturbation
[Qx, ~]     = qr(Jxty,0);
 Jxx        = zeros(obs.Ndata, Nrand);
for i = 1:Nrand
    Jxx(:,i)    = matvec_Jx(model, HI, Qx(:,i))./obs.std;
end

% Jvx = J*Qv = A * T * B'
% J   = J*Qv*Qv'
% J   = A * T * B'*Qv'
 
[A, T, B]  = svd(Jxx); % final svd
 
 dT         = diag(T);
 ind        = find(dT>=tol); % truncate
 U          = A(:,ind);
 V          = Qx*B(:,ind);
 s          = dT(ind);
 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U, s, V] = more_obs(model, obs, prior, HI, tol, nmax)

% Let O*O = Gamma_{obs}^{-0.5}, L*L' = \Gamma_{pr}
% J  = O *F*diag(dxdu) *L
% Jt = L'*diag(dxdu)*F'*O'

 Nrand      = min(nmax, prior.DoF+5);
 
% apply J to param space perturbation
 O          = randn(prior.DoF, Nrand);
%Qu         = matvec_prior_L (prior, O);
 Jxx        = zeros(obs.Ndata, Nrand);
for i = 1:Nrand
    Jxx(:,i)    = matvec_Jx(model, HI, O(:,i))./obs.std;
end
 
% apply J' to data space perturbation
[Qx, ~]     = qr(Jxx,0);
 Jxty       = zeros(prior.DoF, Nrand);
 for i = 1:Nrand
     Jxty(:,i)  = matvec_Jty(model, HI, Qx(:,i)./obs.std);
 end

% Jvty  = J'*Qv = A * T * B'
% J'    = J'*Qv*Qv'
% J'    = A * T * B'*Qv' 
%       = V * T * U'

[A, T, B]  = svd(Jxty); % final svd
 
 dT         = diag(T);
 ind        = find(dT>=tol); % truncate
 V          = A(:,ind);
 U          = Qx*B(:,ind);
 s          = dT(ind);
 
end