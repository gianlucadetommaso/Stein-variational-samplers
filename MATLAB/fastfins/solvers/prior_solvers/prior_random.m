function samples = prior_random(prior, n)
%PRIOR_RANDOM    
% Generates random variables from the prior distribution
%
% Tiangang Cui, 17/Jan/2014

samples = matvec_prior_L(prior, randn(prior.DoF,n)) + repmat(prior.mean_u,1,n);

end
