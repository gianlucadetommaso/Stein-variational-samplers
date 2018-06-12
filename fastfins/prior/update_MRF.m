function prec = update_MRF(FEM, k, sigma, cond)
%UPDATE_MRF
%
% Q u = w, Q = inv(sigma) * (K(cond) + k I)
%
% Tiangang Cui, 09/May/2014

if length(cond) == 3 % for homogeneous case
    temp    = repmat(cond(:)', FEM.Nel, 1);
else
    temp    = cond;
end
temp_K      = FEM.loc_xx(:)*temp(:,1)' + FEM.loc_yy(:)*temp(:, 2)' + FEM.loc_xy(:)*temp(:,3)';
K           = sparse(FEM.ind_i(:), FEM.ind_j(:), temp_K(:), FEM.Nnode, FEM.Nnode);

nor         = gamma(1)/(gamma(2)*(k^2)*(4*pi)); % normalizing const of kernel
tmp         = sqrt(nor)/sigma;                  % std of the covariance

spML        = spdiags(FEM.ML,        0,FEM.Nnode,FEM.Nnode);
%sqspML     = spdiags(FEM.ML.^0.5,   0,FEM.Nnode,FEM.Nnode);
ispML       = spdiags(FEM.ML.^(-1),  0,FEM.Nnode,FEM.Nnode);
isqspML     = spdiags(FEM.ML.^(-0.5),0,FEM.Nnode,FEM.Nnode);

%P  = (K + c) + spML*(k^2);
P           = K + spML*(k^2);
prec.per    = symamd(P); % this is a MatLab function

% prec.R and prec.sca are used for evaluating the 
prec.R      = chol(P(prec.per,prec.per)); % upper triangular cholesky of the P matrix
prec.sca    = sqrt(FEM.ML(prec.per))/tmp; % sqrt of the scale matrix

prec.RQ     = tmp*isqspML*P; % prec.RQ' * prec.RQ = Q, sqrt of the precision
prec.Q      = P*(ispML*(tmp^2))*P;  % precision matrix

end