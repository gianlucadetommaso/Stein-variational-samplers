function prior = init_prior_dist(prior_def, mesh)
%INIT_PRIOR_DIST   
%
% Initializes the prior and relevant transformations for distributed parameters
%
% Tiangang Cui, 19/March/2014

% passing parameters and set default value
if ~isfield(prior_def,'scale')
    corr.scale  = 0.2*eye(2);
else
    corr.scale  = prior_def.scale;
end
if ~isfield(prior_def,'power')
    corr.power  = 2;
else
    corr.power  = prior_def.power;
end
if ~isfield(prior_def,'sigma')
    corr.sigma  = 1;
else
    corr.sigma  = prior_def.sigma;
end
if ~isfield(prior_def,'k')
    corr.k      = 0;
else
    corr.k      = prior_def.k;
end
if ~isfield(prior_def,'cond')
    corr.cond  = [1; 1; 0];
else
    corr.cond  = prior_def.cond;
end
if ~isfield(prior_def,'cov_type')
    corr.type   = 'GP';
else
    corr.type   = prior_def.cov_type;
end

prior.corr      = corr;

switch  corr.type
    case {'GP', 'MRF'}
        prior.type      = 'Dist';
        prior.NP        = mesh.Nnode;
        prior.DoF       = mesh.Nnode;
    case {'KL'}
        prior.type      = 'KL';
        prior.NP        = prior_def.nparam;
        prior.DoF       = prior_def.nparam;
end

% set the prior
switch  corr.type
    case {'GP'}
        prior.cov   = make_prior_GP (mesh, corr.scale, corr.power, corr.sigma);
    case {'MRF'}        
        prior.cov   = make_prior_MRF(mesh, corr.k,     corr.sigma, corr.cond);
    case {'KL'}
        prior.cov   = make_prior_KL(prior_def, corr.sigma);
end

if ~isfield(prior_def,'func')
    prior.func.type = '';
else
    prior.func.type = prior_def.func;
end

switch prior.func.type
    case{'log'}
        if ~isfield(prior_def,'log_thres')
            prior.func.log_thres = 0;
        else
            prior.func.log_thres = prior_def.log_thres;
        end
    case {'erf'}
        prior.func.erf_scale = prior_def.erf_scale;
        prior.func.erf_shift = prior_def.erf_shift;
end

if ~isfield(prior_def,'mean')
    prior.mean_u = zeros(prior.DoF,1);
else
    if length(prior_def.mean) == 1
        prior.mean_u = prior_def.mean*ones(prior.DoF,1);
    else
        prior.mean_u = prior_def.mean;
    end
end

%prior.mean_v     = prior.cov_whitening(prior.mean_u);

prior.mesh      = mesh;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

