% PLOT_SETUP
% plot the problem setup
% Tiangang Cui 17/May/2014

load model_setup
model.sensors   = (model.k+1):model.k:(model.N+1);
prior           = make_prior(model.Tend, model.N);
prior.true_w    = model.true_w;

vmap    = get_map_matlab(model, obs, prior, randn(prior.DoF, 1)); % map 
umap    = matvec_prior_L(prior, vmap);
soln    = forward_solve(model, u2x(prior, umap));
soln_t  = forward_solve(model, u2x(prior, prior.true_w));


t       = linspace(0, model.Tend, model.N+1);
figure
plot(t, soln.G, 'r-', t, soln_t.G, 'k-', model.sensors*model.dt, obs.data, 'bo')


%{
[~,~,~,~,HI]    = minus_log_post(model, obs, prior, vmap);
[U,S,V]         = svd_explicit_WJ(model, obs, prior, HI);
%}