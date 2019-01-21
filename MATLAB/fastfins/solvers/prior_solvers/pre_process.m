function out = pre_process(prior, v, deri_flag)
%PRE_PROCESS
%
% Transform the computational parameters in to the physical parameter x 
% used by the PDE. Then compute the prior information and gradient 
%
% v is associated with prior N(0, I)
%
% Tiangang Cui, 17/Jan/2014

[out.mlp, out.grad_p]   = minus_log_prior( v ); % v ~ N(0, I), v = in
 u                      = matvec_prior_L(prior, v) + prior.mean_u;

if deri_flag
    [out.x, out.dxdu]   = u2x( prior, u );
else
     out.x              = u2x( prior, u );
end

end


% [out.x, out.dxdu, out.dx2du2] = prior.u2x( prior.func, u ); 
% for the full Hessian