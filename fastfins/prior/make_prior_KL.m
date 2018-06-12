function cov = make_prior_KL(prior_def,sigma)

    cov.C    = sigma.^2*eye(prior_def.nparam);
    cov.RC   = chol(cov.C);
    cov.Q    = inv(cov.C);
    cov.type = 'KL';

end