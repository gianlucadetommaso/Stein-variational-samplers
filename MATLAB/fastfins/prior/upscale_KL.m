function prior_KL_to = upscale_KL(prior_KL_from, map, DoF)

prior_KL_to = prior_KL_from;
% set DoF
prior_KL_to.DoF = DoF;
% set the covariance matrix at this level empty
prior_KL_to.cov = {};
% set the mesh to empty
prior_KL_to.mesh = {};
% set the true_x empty
prior_KL_to.true_x = {};
% set these two empty as they are not defined at the upscaled level
% these doesnt matter here
prior_KL_to.chol2w = {};
prior_KL_to.w2chol = {};
prior_KL_to.basis_w = {};
% upscale mean
prior_KL_to.mean_u = full(map*prior_KL_from.mean_u);
% upscale KL mean, here a truncation is used
prior_KL_to.mean_v = prior_KL_from.mean_v(1:DoF);

prior_KL_to.P = full(map*prior_KL_from.P(:, 1:DoF));
prior_KL_to.basis = full(map*prior_KL_from.basis(:, 1:DoF));

end