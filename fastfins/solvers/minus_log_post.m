function [mlpt, mllkd, gmlpt, gmllkd, HI] = minus_log_post(model, obs, prior, v)
%MODEL_SOLVE
%
% solve the forward model
% compute the minus log-likelihood, and minus log-posterior
% solve the adjoint gradient, optional
% assemble the information for evaluating the matvec with PPH, optional
%
% Tiangang Cui, 19/Mar/2014

if nargout > 2
    c2p = pre_process(prior, v, true);
else
    c2p = pre_process(prior, v, false);
end

HI      = forward_solve(model, c2p.x);
misfit  = (HI.d - obs.data)./obs.std;
mllkd   = 0.5*sum(misfit(:).^2); % minus log-likelihood
mlpt    = mllkd + c2p.mlp;       % minus log posterior

if nargout == 3 || nargout == 4
    % gx    = adjoint(model, HI, misfit);
    gx      = matvec_Jty(model, HI, misfit./obs.std);
    gmllkd  = matvec_prior_Lt(prior, c2p.dxdu.*gx);
    gmlpt   = gmllkd + c2p.grad_p;
end

if nargout == 5
    gx      = matvec_Jty(model, HI, misfit./obs.std);
    gmllkd  = matvec_prior_Lt(prior, c2p.dxdu.*gx);
    gmlpt   = gmllkd + c2p.grad_p;
    HI.dxdu = c2p.dxdu;
end

% hessinfo.grad_p = c2p.grad_p;

%}

end