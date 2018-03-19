%% Script conditional diffusion test case
%
% By Gianluca Detommaso 15/03/2018
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

% Set maximum run times
timemax = [100 500 1000];

% Run the algorithms and create plot
figure('name', '100D conditional diffusion')

% SVQN-FI
stepsize = 1;

for j = 1:3   
    x_SVQN_FI = SVQN_FI(N, stepsize, timemax(j), model, prior, obs);

    subplot(4,3,j)
    Eu  = mean(x_SVQN_FI,2);
    Eu2 = mean(x_SVQN_FI.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = Eu + 1.96*sdu;
    lwbound = Eu - 1.96*sdu;
    f1 = fill([model.t'; flipud(model.t')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(model.t, obs.u_true, 'k-')
    for k = 1:obs.nobs
        plot(obs.y_tt, obs.y(:,k), 'r.', 'markersize', 7)
    end
    plot(model.t', Eu, 'b-')
    grid on
    title(['SVQN-FI -- ' num2str(floor(timemax(j))) 's'])
end

% SVQN-I
stepsize = 1;

for j = 1:3   
    x_SVQN_I = SVQN_I(N, stepsize, timemax(j), model, prior, obs);
    
    subplot(4,3,3 + j)
    Eu  = mean(x_SVQN_I,2);
    Eu2 = mean(x_SVQN_I.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = Eu + 1.96*sdu;
    lwbound = Eu - 1.96*sdu;
    f1 = fill([model.t'; flipud(model.t')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(model.t, obs.u_true, 'k-')
    for k = 1:obs.nobs
        plot(obs.y_tt, obs.y(:,k), 'r.', 'markersize', 7)
    end
    plot(model.t', Eu, 'b-')
    grid on    
    title(['SVQN-I -- ' num2str(floor(timemax(j))) 's'])
end

% SVGD-FI
stepsize = 1;

for j = 1:3   
    x_SVGD_FI = SVGD_FI(N, stepsize, timemax(j), model, prior, obs);
    
    subplot(4,3,6 + j)
    Eu  = mean(x_SVGD_FI,2);
    Eu2 = mean(x_SVGD_FI.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = Eu + 1.96*sdu;
    lwbound = Eu - 1.96*sdu;
    f1 = fill([model.t'; flipud(model.t')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(model.t, obs.u_true, 'k-')
    for k = 1:obs.nobs
        plot(obs.y_tt, obs.y(:,k), 'r.', 'markersize', 7)
    end
    plot(model.t', Eu, 'b-')
    grid on
    title(['SVGD-FI -- ' num2str(floor(timemax(j))) 's'])
end

% SVGD-I
stepsize = 1e-3;

for j = 1:3    
    x_SVGD_I = SVGD_I(N, stepsize, timemax(j), model, prior, obs);
    
    subplot(4,3,9 + j)
    Eu  = mean(x_SVGD_I,2);
    Eu2 = mean(x_SVGD_I.^2,2);
    Vu  = Eu2 - Eu.^2;
    sdu = sqrt(Vu);
    upbound = Eu + 1.96*sdu;
    lwbound = Eu - 1.96*sdu;
    f1 = fill([model.t'; flipud(model.t')], [lwbound; flipud(upbound)], 'b'); hold on
    set(f1,'facealpha',.1)
    plot(model.t, obs.u_true, 'k-')
    for k = 1:obs.nobs
        plot(obs.y_tt, obs.y(:,k), 'r.', 'markersize', 7)
    end
    plot(model.t', Eu, 'b-')
    grid on
    title(['SVGD-I -- ' num2str(floor(timemax(j))) 's'])  
end