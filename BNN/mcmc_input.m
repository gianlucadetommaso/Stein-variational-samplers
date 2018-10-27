function [def, out] = mcmc_input(def, user)
%MCMC_INPUT
%   Process the input arguments for MCMC
%
%   Tiangang Cui, 25/April/2014

% initial guess
if ~isfield(def, 'init')
    disp('Error: initial guess needed');
end

% Parameter size
def.np              = length(def.init);

% number of MCMC steps
% if ~isfield(def, 'nstep')
%     def.nstep       = 1E6;
% end

% default batch size and target acceptance rate for adjust MCMC
if ~isfield(def, 'nbatch')
    def.nbatch      = 50;
end

% default acceptance rate
if ~isfield(def, 'rate')
    def.rate        = user.rate;
end

% default jump size
if ~isfield(def, 'sigma')
    def.sigma       = log(1/sqrt(def.np));
end

def.out_size        = 0;

% only use for high dimensional problems
if user.high_d
    % output
    if ~isfield(def, 'save_batch')
        def.save_batch  = 1;
    end
    if isfield(def, 'projection')
        if ~isempty(def.projection)
            if size(def.projection,1) ~= def.np
                disp('Error: wrong projection size')
                return;
            end
        end
        def.out_size    = size(def.projection,2);
    else
        def.projection  = [];
    end
    % save batch
    % if save every thing
    if ~isfield(def, 'save_all')
        def.save_all    = true;
    end
end

% create outputs
if user.high_d
    out.j       = 0;
    out.k       = 0;
    out.size    = floor(def.nstep/def.save_batch);
    if def.save_all
        out.v_samples   = zeros(out.size,def.np);
    end
    if def.out_size > 0
        out.p_samples   = zeros(def.nstep,def.out_size);
    end
    out.acc     = zeros(floor(def.nstep/def.nbatch),2);
else
    % create outputs
    out.k       = 0;
    out.v_samples       = zeros(def.nstep,def.np);
end
out.lpt         = zeros(def.nstep,1);
out.llkd        = zeros(def.nstep,1);
out.sigma       = zeros(floor(def.nstep/def.nbatch),1);
out.mh          = zeros(def.nstep,1);


end