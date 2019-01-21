function dt_out = matvec_Fisher(model, data, HI, dt_in)
% compute the matrix vector product with the Fisher information
% approximation of the Hessian
%
%
% Tiangang Cui, 06/August/2018

dt_out = zeros(size(dt_in));

for i = 1:size(dt_in, 2)
    dy  = matvec_Jx(model, HI, dt_in(1:model.N_w, i));
    dt_out(1:model.N_w, i) = matvec_Jty(model, HI, dy) * HI.gamma;
end

dt_out(1:model.N_w, :) = dt_out(1:model.N_w, :) + dt_in(1:model.N_w, :) * HI.lambda;

if model.hyper_on
    dt_out(model.ind_log_gamma, :)  = (0.5*data.N_y  + model.beta_gamma*HI.gamma)   * dt_in(model.ind_log_gamma, :);
    dt_out(model.ind_log_lambda, :) = (0.5*model.N_w + model.beta_lambda*HI.lambda) * dt_in(model.ind_log_lambda, :);
else
    dt_out(model.ind_log_gamma, :)  = dt_in(model.ind_log_gamma, :);
    dt_out(model.ind_log_lambda, :) = dt_in(model.ind_log_lambda, :);
end

end