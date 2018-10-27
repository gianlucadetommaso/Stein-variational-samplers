function out = hmala(mcmc_def)
%HMALA
%
% Hessian preconditioned Metropolis adjusted Langevin MCMC sampler
%
% Tiangang Cui, 17/Jan/2014
%
% Reference:
% Martin, J., Wilcox, L. C., Burstedde, C., & Ghattas, O. (2012). A stocha-
% stic Newton MCMC method for large-scale statistical inverse problems with
% application to seismic inversion. SIAM Journal on Scientific Computing,
% 34(3), A1460-A1487.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Tuning parameters are given by mcmc_def structure, it has the following
% inputs
%
% nstep:
%       Number of MCMC steps, default is 10,000.
%
% nabtch:
%       Number of batch size used for adaptive changing the mcmc jump size
%       and empirical estimation of the covariance, default is 50.
%
% rate:
%       Target accceptance rate, default is 0.58
%
% sigma:
%       exp(sigma) for the proposal step size
%
% V:
%       Eigenvectors of the Hessian at the MAP
%
% d:
%       Eigenvalues of the Hessian at the MAP
%
% Options for saving outputs:
%
% save_bacth:
%       The batch size for saving mcmc history, default is 1.
%
% projection:
%       A set of vectors for projecting the MCMC sample trace of the
%       parameters, default is [], for memory saving.
%
% save_all:
%       A flag indicates save the entire MCMC trace of parameters, default
%       is false. Only turn on with sufficient amount of memory
%
% Pass-in function required:
%
% [minus_log_post, minus_log_likelihood, grad] = minus_log_post(v)
%
% In the examples we have, the following is used
% minus_log_post = @(v) minus_log_post_mcmc(mesh, obs, prior, param, FEM, v)
%
% input:
%       v, parameters equipped with identity prior covariance
%
% output:
%       minus_log_post, minus_log_likelihood, and grad, where grad is the
%       gradient of minus_log_post w.r.t. v
%
% This function is passed-in by the structure mcmc_def, e.g.,
% mcmc_def.minus_log_post = @(v) minus_log_post_mcmc(..., v);
% where ... are used provided data structures for setup the forward
% simulation, and evaluating posterior distribution, this function only
% pass-in one variable v.
%
% Note on the the Hessian:
%
% We use the computational parameters v = inv_sqrt(C)u, all the operations
% are defined on the computational parameters.
%
% Let H denote the Hessian of the llkd, and C the prior covariance, then
% V diag(d) V'= C^{0.5} H C^{0.5}.
%
% In the examples supplied, V and d are given by:
% [V, d] = generalized_eigen(mesh, obs, prior, param, FEM, y, 1E-5, ...
%               ceil(obs.N_obs*3));
% User has to supplied their own version for different models
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[def, out]  = process_input(mcmc_def); % parse inputs
v_curr      = def.init;
lap         = get_laplace(def.V, def.d);
sigma       = def.sigma;

[mlpt_curr, mg_curr] = def.minus_log_post(v_curr / mcmc_def.lambdasqrt);
mg_curr = mg_curr / mcmc_def.lambdasqrt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if def.lis_flag
    t1          = mg_curr - v_curr;
    out.cross_g = t1*t1';
    out.cross_v = v_curr*v_curr';
    out.sum_v   = v_curr;
    out.nsample = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% start MCMC
acc         = 0;
batch       = 0;

for      i  = 1:def.nstep
    
    if mod(i,100) == 0
        fprintf('Iteration: %d\n', i)
    end
    
    r_curr  = randn(def.np-2,1);
    v_next  = laplace_propose(lap, exp(sigma), v_curr, mg_curr, r_curr);
    
    [mlpt_next, mg_next] = mcmc_def.minus_log_post(v_next / mcmc_def.lambdasqrt);
    mg_next = mg_next / mcmc_def.lambdasqrt;
    
    r_next  = laplace_reverse(lap, exp(sigma), mg_curr, mg_next, r_curr);
    log_n2c = - 0.5 * (r_next'*r_next);
    log_c2n = - 0.5 * (r_curr'*r_curr);
    alpha   = (mlpt_curr - mlpt_next) + (log_n2c - log_c2n); % log acceptance prob.
    
    if log(rand)    < alpha
        v_curr      = v_next;
        mg_curr     = mg_next;
        mlpt_curr   = mlpt_next;
        acc         = acc+1;
    end
    
    batch   = batch + 1;
    
    if  batch       == def.nbatch
        delta       = min(0.1,sqrt(def.nbatch/i));
        if (acc/def.nbatch) < def.rate
            sigma       = sigma - delta;
        else
            sigma       = sigma + delta;
        end
        disp(sigma)
        batch               = 0;
        acc                 = 0;
        out.k               = out.k+1;
        out.sigma(out.k)    = sigma;
        out.acc(out.k,1)    = acc/def.nbatch;
    end
    
    if def.lis_flag
        t1          = mg_curr - v_curr;
        out.cross_g = out.cross_g + t1*t1';
        out.cross_v = out.cross_v + v_curr*v_curr';
        out.sum_v   = out.sum_v + v_curr;
        out.nsample = out.nsample+1;
    end
    
    % save    
    if def.save_all && mod(i, def.save_batch) == 0
        out.j                   = out.j + 1;
        out.v_samples(out.j,:)  = v_curr;
    end
    if def.out_size > 0
        out.p_samples(i,:)  = def.projection'*v_curr;
    end
    out.lpt (i)             = mlpt_curr;
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function laplace = get_laplace(V, d)

% w.r.t. llkd
ipseudo         = (d + 1).^(-1) - 1;
ipseudo_half    = (d + 1).^(-0.5) - 1;
laplace.V       = V;

laplace.V_ihalf = scale_cols( V, ipseudo_half ); % V*( (Lambda+I)^(-0.5) - I ), for random number
laplace.V_half  = scale_cols( V, sqrt(d) ); % V*Lambda^(0.5), for density
laplace.V_iH    = scale_cols( V, ipseudo ); % V*( (Lambda+I)^(-1) - I ), for scale the gradient
% The determinant
% laplace.log_det_H_half = log(abs( prod((d + 1).^(0.5)) * ( model.param.NC*prod(diag(model.param.P)) ) ));

%laplace.H_half = scale_cols(V, sqrt(ipseudo));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = laplace_propose(lap, scale, x, gx, r)
% Chop hyperparameter parts
gx = gx(1:end-2);
a = -(0.5*scale^2)*gx;
y = x(1:end-2) + (a + lap.V_iH*(lap.V'*a)) + scale*(r + lap.V_ihalf*(lap.V'*r));
y = [y; x(end-1:end)];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function w = laplace_reverse(lap, scale, gx, gy, r)
% Chope hyperparameter parts
gx = gx(1:end-2);
gy = gy(1:end-2);
a = (gx+gy);
w = (0.5*scale)*(a + lap.V_ihalf*(lap.V'*a)) - r;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [def, out] = process_input(def)

user.rate       = 0.58;
user.high_d     = true;
[def, out]      = mcmc_input(def, user);

% default flag for building the LIS using gradient
if isfield(def, 'build_lis')
    def.lis_flag    = true;
else
    def.lis_flag    = false;
end

% check the laplace strcut
if ~isfield(def, 'V') || ~isfield(def, 'd')
    disp('Error: Need to compute the Laplace approximation!');
    return;
end

end