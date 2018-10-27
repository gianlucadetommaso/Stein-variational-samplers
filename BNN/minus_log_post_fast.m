function mlpt = minus_log_post_fast(model, data, theta)
%minus_log_post
%
% solve the forward model
% compute the minus log-likelihood, and minus log-posterior, without the
% hyper-parameter part
% solve the adjoint gradient, optional
% assemble the information for evaluating the matvec with PPH, optional
%
% Tiangang Cui, 03/August/2018

w           = theta(1:model.N_w, 1);
t_gamma     = theta(model.ind_log_gamma, 1);
t_lambda    = theta(model.ind_log_lambda, 1);

HI          = NN_model_fast(model, data.xs, w);

HI.gamma    = exp(t_gamma);
HI.lambda   = exp(t_lambda);

% data misfit 
misfit  = (HI.zs{end} - data.ys); 
% minus log-likelihood
mllkd   = 0.5 * norm(misfit(:))^2 * HI.gamma - 0.5 * t_gamma * data.N_y; 
% minus log-prior
mlp     = 0.5 * norm(w)^2 * HI.lambda - 0.5 * t_lambda * model.N_w; 
% minus log-hyper
mlhp    =   - model.alpha_gamma * t_gamma   + model.beta_gamma * HI.gamma ...
            - model.alpha_lambda * t_lambda + model.beta_lambda * HI.lambda;
% minus log posterior
mlpt    = mllkd + mlp + mlhp;      

end