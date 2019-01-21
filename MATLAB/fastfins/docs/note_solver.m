%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Avoid using function handles, as it has unclear impact on the memory
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data structures:
%
%*************************************
%*************************************
%   model:  A data structure contains a collection of data for evaluating
%           the forward model, the adjoint gradient, and the Hessian. This 
%           data structure is problem dependent.
%
%*************************************
%           It needs the following user-supplied functions:
%
%               forward(model, x)
%                   forward model   -- mandatory
%
%               matvec_Jty(model, HI, dy)
%                   matvec of Jacobian transpose, adjoint model
%
%               matvec_Jx (model, HI, dx)
%                   matvec of Jacobian, linearized forward model
%
%               explicit_J(model, HI)
%                   Explicit Jacobian, only avalable for stationary problems
%
%*************************************
%           The data structure must has the following
%
%           model.Nsensors
%
%           model.Ndatasets
%
%           model.mesh
%
%           model.SVD
%
%*************************************
%           These two functions are no longer needed
%
%               adjoint(model, HI, y)
%                   adjoint model   -- for gradient
%
%               matvec_GNH(model, HI, dx) 
%                   matrix vector product (matvec) of Gaussian-Newton Hessian 
%
%
%*************************************
%*************************************
%   obs:    
%           obs.data
%           obs.Nsensors
%           obs.Ndatasets
%           obs.Ndata
%           obs.std
%
%*************************************
%*************************************
%   prior:  A data structure describes the prior mean and covairance. 
%           It also conatins the transformation between the Gaussian random
%           variable and the physical parameters (model inputs).
% 
%*************************************
%               For transform between the Gaussian variable u and the physical
%               variable x
%
%               prior.func
%                   a indicator for the type of transformation
%
%               u2x(prior, u) 
%                   convert the Gaussian variable u to the physical
%                   variable x
%
%               for the option ``log''
%                   x = exp(u) + prior.log_thres
%
%               for the option ``erf''
%                   x = erf(u).*prior.erf_scale + prior.erf_shift;
%               
%*************************************
%               For manipulating with prior covariance, let prior = N(m, C)
%               and C = L*L'
% 
%               cov_invLu (cov, u)      -- inv(L) *u        
%
%               cov_invLtu(cov, u)      -- inv(Lt)*u        
%                   
%               cov_Ltv(cov, v)         -- L'*v
%
%               cov_Lv (cov, v)         -- L *v
%
%*************************************
%           The data structure must has the following
%
%               prior.type
%
%               prior.DoF
%
%               prior.cov
%
%*************************************
%*************************************
%   v:      The computaional parameter, associated with prior N(0, I). 
%           All the optimization algorithms and MCMC algorithms are dealing
%           with this ``v'' parameter. See notes_paramemter.m for details.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   minus_log_post:     Evaluating the minus log posterior density.
%
%           Inputs:     model, obs, prior, v
%
%           Outputs:    
%               mlpt    -- minus log posterior  
%               mllkd   -- minus log likelihood
%               gmlpt   -- gradient of mlpt 
%               gmllkd  -- gradient of mllkd
%               HI      -- for computing the Hessian matvec         
%
%   matvec_PPGNH:       matvec with prioir preconditioned Hessian
%           
%           Inputs:     model, obs, prior, HI, dv
%
%   eigen_PPGNH:        eigendecomposition of the PPGNH
%
%           Inputs:     model, obs, prior, HI, tol, Nmax
%
%   svd_rand_WJ:        randomized svd of the linearized forward model
%                       after whitening transformation on both side
%
%           Inputs:     model, obs, prior, HI, tol, Nmax
%
%   get_map_matlab:     find the MAP using matlab fminunc
%
%           Inputs:     model, obs, prior, HI
%
%   svd_explicit_WJ:    explicit Jacobian
%
%           Inputs:     model, obs, prior, HI
%
%*************************************
%*************************************
%
%   prior_solver:       Function used for handling the prior structure
%
%   pre_process:
%
%           Evaluating the prior density, and relevant transformation for
%           the changing of variable between computational parameter and
%           the physical parameter
%
%           Inputs:     prior, v, flag
%                       flag is for indicating if the gradient or Hessian
%                       is needed
%
%           Outputs:    the physical parameter x, minus log prior density
%                       if flag == true, also returns the prior gradient and
%                       the Jacobian of the change of the transformation from 
%                       v to x
%
%   u2x:                transform from u to physical parameter x, u ~ N(m, C) 
%
%           Inputs:     prior, u
%
%   minus_log_prior:    minus log prior and gradient
% 
%   matvec_prior_invL:  Whitening transformation, inv(L)*u
%
%           Inputs:     prior, u 
%
%   matvec_prior_invLt: inv(L')*u
%
%           Inputs:     prior, u 
%
%   matvec_prior_L:     L*v
%
%           Inputs:     prior, v 
%
%   matvec_prior_Lt:    Lt*v
%
%           Inputs:     prior, v 
%
%   basis_LIS:          build oblique basis for the LIS
%           
%           Inputs:     prior, P
%
%   KL_basis:           build the prior KL basis 
%
%           Inputs:     prior, thres
%
%   prior_random:       random prior samples
%           
%           Inputs:     prior, n
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%