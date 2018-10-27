%% Script conditional diffusion test case
%
% By Gianluca Detommaso -- 8/06/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Set random seed
rng(1);

% Load directories
load_dir

% Run model and plot setup
plot_setup

% Number of particles
npart = 1e2;

% Initial particle configuration
w_init = randn(model.N, npart);  

% Number of iterations
itermax = 10;

% Estimate computational time
stepsize = 1;
[~, ~, t_NH]  = SVN_H(w_init, stepsize, itermax, model, obs);

stepsize = 1;
[~, ~, t_NI]  = SVN_I(w_init, stepsize, itermax, model, obs);

stepsize = 1e-1;
[~, ~, t_GDH] = SVGD_H(w_init, stepsize, itermax, model, obs);

stepsize = 5*1e-3;
[~, ~, t_GDI] = SVGD_I(w_init, stepsize, itermax, model, obs);

% Time ratios wrt SVN_H
r_NI   = t_NH / t_NI;
r_GDH  = t_NH / t_GDH;
r_GDI  = t_NH / t_GDI;

% Traceplots figure
figure('name', '100D conditional diffusion')

% Time steps and observation times
tt = 0:model.dt:model.Tend;
tt_obs = linspace(model.dt*model.k, model.Tend, obs.Ndata);

% Calculate true solution
sol_true = euler_solve(model.N, model.d, model.dt, model.true_w);

    
%% Compare the algorithms with same total cost

% SVN-H
itermax_NH     = [10 50 100];
itermaxdiff_NH = [itermax_NH(1) diff(itermax_NH)];
stepsize_NH    = 1;  
w_NH           = w_init;

for j = 1:3   
    % Run SVN-H
    [w_NH, stepsize_NH] = SVN_H(w_NH, stepsize_NH, itermaxdiff_NH(j), model, obs);

    % Calculate solution
    sol_NH = zeros(model.N+1, npart);
    for k = 1:npart
        sol_NH(:,k) = euler_solve(model.N, model.d, model.dt, w_NH(:,k));
    end

    % Plot reconstructed path and 90% confidence interval
    subplot(4,3,j)
    Eu  = mean(sol_NH,2);
    Eu2 = mean(sol_NH.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = quantile(sol_NH,0.95,2);
    lwbound = quantile(sol_NH,0.05,2);
    f1 = fill([tt'; flipud(tt')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(tt, sol_true, 'm-')
    plot(tt_obs', obs.data, 'r.', 'markersize', 10)
    plot(tt', Eu, 'b-')
    grid on
    title(['SVN-H -- ' num2str(floor(itermaw_NH(j))) ' iterations'])
end


% SVN-I
itermax_NI     = ceil(r_NI*itermaw_NH);
itermaxdiff_NI = [itermax_NI(1) diff(itermax_NI)];
stepsize_NI    = 1;  
w_NI           = w_init;

for j = 1:3   
    % Run SVN-I
    [w_NI, stepsize_NI] = SVN_I(w_NI, stepsize_NI, itermaxdiff_NI(j), model, obs);
    
    % Calculate solution
    sol_NI = zeros(model.N+1, npart);
    for k = 1:npart
        sol_NI(:,k) = euler_solve(model.N, model.d, model.dt, w_NI(:,k));
    end

    % Plot reconstructed path and 90% confidence interval
    subplot(4,3,3 + j)
    Eu  = mean(sol_NI,2);
    Eu2 = mean(sol_NI.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = quantile(sol_NI,0.95,2);
    lwbound = quantile(sol_NI,0.05,2);
    f1 = fill([tt'; flipud(tt')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(tt, sol_true, 'm-')
    plot(tt_obs', obs.data, 'r.', 'markersize', 10)
    plot(tt', Eu, 'b-')
    grid on
    title(['SVN-I -- ' num2str(floor(itermax_NI(j))) ' iterations'])
end

% SVGD-H
itermax_GDH     = ceil(r_GDH*itermaw_NH); 
itermaxdiff_GDH = [itermax_GDH(1) diff(itermax_GDH)];
stepsize_GDH    = 1e-1;
w_GDH           = w_init;

for j = 1:3  
    % Run SVGD-H
    [w_GDH, stepsize_GDH] = SVGD_H(w_GDH, stepsize_GDH, itermaxdiff_GDH(j), model, obs);
   
    % Calculate solution
    sol_GDH = zeros(model.N+1, npart);
    for k = 1:npart
        sol_GDH(:,k) = euler_solve(model.N, model.d, model.dt, w_GDH(:,k));
    end

    % Plot reconstructed path and 90% confidence interval
    subplot(4,3,6 + j)
    Eu  = mean(sol_GDH,2);
    Eu2 = mean(sol_GDH.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = quantile(sol_GDH,0.95,2);
    lwbound = quantile(sol_GDH,0.05,2);
    f1 = fill([tt'; flipud(tt')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(tt_obs', obs.data, 'r.', 'markersize', 10)
    plot(tt', Eu, 'b-')
    grid on
    title(['SVGD-H -- ' num2str(floor(itermax_GDH(j))) ' iterations'])
end

% SVGD-I
itermax_GDI     = ceil(r_GDI*itermaw_NH); 
itermaxdiff_GDI = [itermax_GDI(1) diff(itermax_GDI)];
stepsize_GDI    = 5*1e-3;
w_GDI           = w_init;

for j = 1:3 
    % Run SVGD-I
    [w_GDI, stepsize_GDI] = SVGD_I(w_GDI, stepsize_GDI, itermaxdiff_GDI(j), model, obs);
   
    % Calculate solution
    sol_GDI = zeros(model.N+1,npart);
    for k = 1:npart
        sol_GDI(:,k) = euler_solve(model.N, model.d, model.dt, w_GDI(:,k));
    end

    % Plot reconstructed path and 90% confidence interval
    subplot(4,3,9 + j)
    Eu  = mean(sol_GDI,2);
    Eu2 = mean(sol_GDI.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = quantile(sol_GDI,0.95,2);
    lwbound = quantile(sol_GDI,0.05,2);
    f1 = fill([tt'; flipud(tt')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(tt, sol_true, 'm-')
    plot(tt_obs', obs.data, 'r.', 'markersize', 10)
    plot(tt', Eu, 'b-')
    grid on
    title(['SVGD-I -- ' num2str(floor(itermax_GDI(j))) ' iterations'])
end


%% Compare SVN-H with DILI

% Set up and run HMC
run_dili_lis
w_DILI = out_dili.v_samples';

% Calculate solution
sol_DILI = zeros(model.N+1, npart);
for k = 1:out_dili.size
    sol_DILI(:,k) = euler_solve(model.N, model.d, model.dt, w_DILI(:,k));
end

% Calculate maximum componentwise integrated autocorrelation time
tau = 0;
for i = 1:model.N
    tau = max(tau, iact(out_dili.v_samples(:,i)));
end
tau = ceil(tau);

% Set burnin
brn = 200;

% Comparison plot
figure('name', 'Comparison SVN-H vs. MCMC')

upboundNH  = quantile(sol_NH, 0.95, 2);
lwboundNH  = quantile(sol_NH, 0.05, 2);
upboundDILI = quantile(sol_DILI(:,brn:tau:end), 0.95, 2);
lwboundDILI = quantile(sol_DILI(:,brn:tau:end), 0.05, 2);

plot(tt, sol_true, 'm-', 'LineWidth', 3)
hold on
plot(tt, mean(sol_NH,2), 'b-', 'LineWidth', 3)
plot(tt, mean(sol_DILI(:,brn:tau:end),2), 'g-', 'linewidth', 3)
plot(tt_obs', obs.data, 'r.', 'markersize', 10)


fDILI = fill([tt'; flipud(tt')], [lwboundDILI; flipud(upboundDILI)], 'g'); hold on
set(fDILI,'facealpha',.1)
fNH  = fill([tt'; flipud(tt')], [lwboundNH; flipud(upboundNH)], 'b'); hold on
set(fNH,'facealpha',.1)
grid on

legend('true', 'SVN-H', 'MCMC', 'obs')