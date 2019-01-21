function prior = build_collocated_KL(prior_in, efl, ev, DoF)
%BASIS_KL
% 
% Tiangang Cui, 17/Jan/2014

prior           = prior_in;
prior.DoF       = DoF;
prior.P         = efl(:,1:DoF);
S               = ev(1:DoF).^(0.5);
prior.basis     = scale_cols(prior.P, S);
prior.basis_w   = {};
prior.type      = 'Basis';
prior.mean_v    = zeros(prior.DoF, 1);
prior.note      = 'KL';

%figure
%semilogy(cumsum(d)/sum(d))
%title('Prior cummulative energy')

end