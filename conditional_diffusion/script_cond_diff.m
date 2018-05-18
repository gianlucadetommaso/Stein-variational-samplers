%% Script conditional diffusion test case
%
% By Gianluca Detommaso 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Turn on parallel pool if not on already
pool = gcp('nocreate');
if isempty(pool)
    parpool;
end

% Set random seed
rng(1);

% Load directories
load_dir_cond_diff;

% Set up the model
setup;

% Number of particles
N = 1e3;

% Initial particle configuration
x_init = prior.m0 + prior.C0sqrt*randn(model.n,N);  

% Number of iterations
itermax = 1000;

% Estimate computational time
stepsize = 1;
[~, ~, t_NH]  = SVN_H(x_init, stepsize, itermax, model, prior, obs);

stepsize = 1;
[~, ~, t_NI]  = SVN_I(x_init, stepsize, itermax, model, prior, obs);

stepsize = 1e-1;
[~, ~, t_GDH] = SVGD_H(x_init, stepsize, itermax, model, prior, obs);

stepsize = 5*1e-3;
[~, ~, t_GDI] = SVGD_I(x_init, stepsize, itermax, model, prior, obs);

% Time ratios wrt SVN_H
r_NI   = t_NH / t_NI;
r_GDH  = t_NH / t_GDH;
r_GDI  = t_NH / t_GDI;

% Traceplots figure
figure('name', '100D conditional diffusion')


%% Compare the algorithms with same total cost

% SVN-H
itermax_NH     = [10 50 100];
itermaxdiff_NH = [itermax_NH(1) diff(itermax_NH)];
stepsize_NH    = 1;  
x_NH           = x_init;

for j = 1:3   
    % Run SVN-H
    [x_NH, stepsize_NH] = SVN_H(x_NH, stepsize_NH, itermaxdiff_NH(j), model, prior, obs);

    % Calculate solution
    sol_NH = zeros(model.n,N);
    for k = 1:N
        sol_NH(:,k) = [x_NH(1,k); zeros(model.n-1,1)];
        for i = 2:model.n
            sol_NH(i,k) = sol_NH(i-1,k)*( 1 + model.beta*( 1-sol_NH(i-1,k)^2 ) ...
                / ( 1+sol_NH(i-1,k)^2 ) * model.h ) + (x_NH(i,k) - x_NH(i-1,k));   
        end
    end

    % Plot reconstructed path and 90% confidence interval
    subplot(4,3,j)
    Eu  = mean(sol_NH,2);
    Eu2 = mean(sol_NH.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = quantile(sol_NH,0.95,2);
    lwbound = quantile(sol_NH,0.05,2);
    f1 = fill([model.t'; flipud(model.t')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(model.t, model.sol_true, 'm-')
    for k = 1:obs.nobs
        plot(obs.t, obs.y(:,k), 'r.', 'markersize', 5)
    end
    plot(model.t', Eu, 'b-')
    grid on
    title(['SVN-H -- ' num2str(floor(itermax_NH(j))) ' iterations'])
end

% SVN-I
itermax_NI     = ceil(r_NI*itermax_NH);
itermaxdiff_NI = [itermax_NI(1) diff(itermax_NI)];
stepsize_NI    = 1;  
x_NI           = x_init;

for j = 1:3   
    % Run SVN-I
    [x_NI, stepsize_NI] = SVN_I(x_NI, stepsize_NI, itermaxdiff_NI(j), model, prior, obs);
    
    % Calculate solution
    sol_NI = zeros(model.n,N);
    for k = 1:N
        sol_NI(:,k) = [x_NI(1,k); zeros(model.n-1,1)];
        for i = 2:model.n
            sol_NI(i,k) = sol_NI(i-1,k)*( 1 + model.beta*( 1-sol_NI(i-1,k)^2 ) ...
                / ( 1+sol_NI(i-1,k)^2 ) * model.h ) + (x_NI(i,k) - x_NI(i-1,k));   
        end
    end

    % Plot reconstructed path and 90% confidence interval
    subplot(4,3,3 + j)
    Eu  = mean(sol_NI,2);
    Eu2 = mean(sol_NI.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = quantile(sol_NI,0.95,2);
    lwbound = quantile(sol_NI,0.05,2);
    f1 = fill([model.t'; flipud(model.t')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(model.t, model.sol_true, 'm-')
    for k = 1:obs.nobs
        plot(obs.t, obs.y(:,k), 'r.', 'markersize', 5)
    end
    plot(model.t', Eu, 'b-')
    grid on
    title(['SVN-I -- ' num2str(floor(itermax_NI(j))) ' iterations'])
end

% SVGD-H
itermax_GDH     = ceil(r_GDH*itermax_NH); 
itermaxdiff_GDH = [itermax_GDH(1) diff(itermax_GDH)];
stepsize_GDH    = 1e-1;
x_GDH           = x_init;

for j = 1:3  
    % Run SVGD-H
    [x_GDH, stepsize_GDH] = SVGD_H(x_GDH, stepsize_GDH, itermaxdiff_GDH(j), model, prior, obs);
   
    % Calculate solution
    sol_GDH = zeros(model.n,N);
    for k = 1:N
        sol_GDH(:,k) = [x_GDH(1,k); zeros(model.n-1,1)];
        for i = 2:model.n
            sol_GDH(i,k) = sol_GDH(i-1,k)*( 1 + model.beta*( 1-sol_GDH(i-1,k)^2 ) ...
                / ( 1+sol_GDH(i-1,k)^2 ) * model.h ) + (x_GDH(i,k) - x_GDH(i-1,k));   
        end
    end

    % Plot reconstructed path and 90% confidence interval
    subplot(4,3,6 + j)
    Eu  = mean(sol_GDH,2);
    Eu2 = mean(sol_GDH.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = quantile(sol_GDH,0.95,2);
    lwbound = quantile(sol_GDH,0.05,2);
    f1 = fill([model.t'; flipud(model.t')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(model.t, model.sol_true, 'm-')
    for k = 1:obs.nobs
        plot(obs.t, obs.y(:,k), 'r.', 'markersize', 5)
    end
    plot(model.t', Eu, 'b-')
    grid on
    title(['SVGD-H -- ' num2str(floor(itermax_GDH(j))) ' iterations'])
end

% SVGD-I
itermax_GDI     = ceil(r_GDI*itermax_NH); 
itermaxdiff_GDI = [itermax_GDI(1) diff(itermax_GDI)];
stepsize_GDI    = 5*1e-3;
x_GDI           = x_init;

for j = 1:3 
    % Run SVGD-I
    [x_GDI, stepsize_GDI] = SVGD_I(x_GDI, stepsize_GDI, itermaxdiff_GDI(j), model, prior, obs);
   
    % Calculate solution
    sol_GDI = zeros(model.n,N);
    for k = 1:N
        sol_GDI(:,k) = [x_GDI(1,k); zeros(model.n-1,1)];
        for i = 2:model.n
            sol_GDI(i,k) = sol_GDI(i-1,k)*( 1 + model.beta*( 1-sol_GDI(i-1,k)^2 ) ...
                / ( 1+sol_GDI(i-1,k)^2 ) * model.h ) + (x_GDI(i,k) - x_GDI(i-1,k));   
        end
    end

    % Plot reconstructed path and 90% confidence interval
    subplot(4,3,9 + j)
    Eu  = mean(sol_GDI,2);
    Eu2 = mean(sol_GDI.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = quantile(sol_GDI,0.95,2);
    lwbound = quantile(sol_GDI,0.05,2);
    f1 = fill([model.t'; flipud(model.t')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(model.t, model.sol_true, 'm-')
    for k = 1:obs.nobs
        plot(obs.t, obs.y(:,k), 'r.', 'markersize', 5)
    end
    plot(model.t', Eu, 'b-')
    grid on
    title(['SVGD-I -- ' num2str(floor(itermax_GDI(j))) ' iterations'])
end


%% Compare SVN-H with Hamiltonian Monte Carlo (HMC)

% Set up and run HMC
hmc   = setup_hmc(model, prior);
out   = hmc_sampler(model, prior, obs, hmc);
x_HMC = out.x;

% Calculate solution
sol_HMC = zeros(model.n,N);
for k = 1:N
    sol_HMC(:,k) = [x_HMC(1,k); zeros(model.n-1,1)];
    for i = 2:model.n
        sol_HMC(i,k) = sol_HMC(i-1,k)*( 1 + model.beta*( 1-sol_HMC(i-1,k)^2 ) ...
            / ( 1+sol_HMC(i-1,k)^2 ) * model.h ) + (x_HMC(i,k) - x_HMC(i-1,k));   
    end
end

% Comparison plot
figure('name', 'Comparison SVN-H vs. HMC')

upboundNH  = quantile(sol_NH,0.95,2);
lwboundNH  = quantile(sol_NH,0.05,2);
upboundHMC = quantile(sol_HMC(:,200:end),0.95,2);
lwboundHMC = quantile(sol_HMC(:,200:end),0.05,2);

plot(model.t, model.sol_true, 'm-', 'LineWidth', 3)
hold on
plot(model.t, mean(sol_NH,2), 'b-', 'LineWidth', 3)
plot(model.t, mean(sol_HMC(:,200:end),2), 'g-', 'linewidth', 3)
for k = 1:obs.nobs
    plot(obs.t, obs.y(:,k), 'r.', 'markersize', 10)
end

fHMC = fill([model.t'; flipud(model.t')], [lwboundHMC; flipud(upboundHMC)], 'g'); hold on
set(fHMC,'facealpha',.1)
fNH  = fill([model.t'; flipud(model.t')], [lwboundNH; flipud(upboundNH)], 'b'); hold on
set(fNH,'facealpha',.1)
grid on

legend('true', 'SVN-H', 'HMC', 'obs')


%% Compare SVGD-I with only N=10 partiles and HMC
figure('name', 'Comparison SVGD-I vs. HMC')

upboundGDI = quantile(sol_GDI,0.95,2);
lwboundGDI = quantile(sol_GDI,0.05,2);
upboundHMC = quantile(sol_HMC(:,200:end),0.95,2);
lwboundHMC = quantile(sol_HMC(:,200:end),0.05,2);

plot(model.t, model.sol_true, 'm-', 'LineWidth', 3)
hold on
plot(model.t, mean(sol_GDI,2), 'b-', 'LineWidth', 3)
plot(model.t, mean(sol_HMC(:,200:end),2), 'g-', 'linewidth', 3)
for k = 1:obs.nobs
    plot(obs.t, obs.y(:,k), 'r.', 'markersize', 10)
end

fHMC = fill([model.t'; flipud(model.t')], [lwboundHMC; flipud(upboundHMC)], 'g'); hold on
set(fHMC,'facealpha',.1)
fGDI  = fill([model.t'; flipud(model.t')], [lwboundGDI; flipud(upboundGDI)], 'b'); hold on
set(fGDI,'facealpha',.1)
grid on

legend('true', 'SVGD-I', 'HMC', 'obs')