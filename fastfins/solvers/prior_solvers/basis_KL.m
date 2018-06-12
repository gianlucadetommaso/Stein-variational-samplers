function prior = basis_KL(prior_in, thres)
%BASIS_KL
% 
% Tiangang Cui, 17/Jan/2014

prior           = prior_in;
[V, d]          = cov_eig(prior_in.cov);

if thres < 1
    jnd         = cumsum(d)/sum(d) <= thres; % truncation 
    prior.DoF   = sum(jnd);
else
    jnd         = 1:floor(thres);
    prior.DoF   = floor(thres);
end
prior.P         = V(:, jnd);
S               = d(jnd).^(0.5);
prior.basis     = scale_cols(prior.P, S);
prior.basis_w   = scale_cols(prior.P, S.^(-1));
prior.type      = 'Basis';
prior.mean_v    = zeros(prior.DoF, 1);
prior.note      = 'KL';
prior.d         = d;
prior.chol2w    = matvec_prior_Lt(prior_in,prior.basis_w);
prior.w2chol    = matvec_prior_invL(prior_in, prior.basis);

%figure
%semilogy(cumsum(d)/sum(d))
%title('Prior cummulative energy')

end