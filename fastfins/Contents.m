% FastFInS 0.9 - fast forward and inverse sovler. pre-release                   
% Author: Tiangang Cui (tcui@mit.edu, tcui001@gmail.com), 01/Oct/2013
%
% Files:
%   initialize                           - 
%           Initialization script for the FEM model
%   note_parameter                       - 
%           A note on the definition of various parameters and relevant
%           transformations
%   readme                               - 
%           The readme file, you won't expect anything fancy here, right?
%
% Folders:
%
% fem_models:
%           Set of forward models
%
% library:
%           Utility functions
%
% mesh_2d:
%           2D regular mesh generator
% 
% optimizer:
%           Subspace trust region reflective solver, not very stable, use
%           the MATLAB sovler if you have it
% 
% sampler:
%
% These three samplers are standard samplers designed for low dimensional 
% problems
%
%   sampler/adaptive/adaptive_mala       - 
%           Metropolis adjusted Langevin, preconditioned using the 
%           empirical covariance, step size is automatically tuned to match
%           acceptance rate 0.58
%
%   sampler/adaptive/adaptive_metropolis - 
%           Metropolis, proposal is the sscaled empirical covariance, step
%           size is automatically tuned to match the acceptance rate 0.23
%
%   sampler/adaptive/amwg                - 
%           Adaptive Metropolis Within Gibbs
%
% Dimension independent, likelihood informed sampler
%
%   sampler/dili_mcmc/dili_mcmc          - 
%           The sampler           
%
%   sampler/dili_mcmc/plot_redu          - 
%           A plot function to plot the subspace
%
%   sampler/h_langevin                   - 
%           Hessian preconditioned Langevin
%
%   sampler/pcn_mcmc                     - 
%           Standard PCN sampler, non-likelihood informed
%
%   sampler/rto_mcmc                     - 
%           Randomize then optimize sampler
%
%