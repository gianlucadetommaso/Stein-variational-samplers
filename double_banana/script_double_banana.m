%% Script double banana test case
%
% By Gianluca Detommaso 15/03/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Turn on parallel pool if not on aoready
pool = gcp('nocreate');
if isempty(pool)
    parpool;
end

% Set random seed
rng(1);

% Load directories
load_dir_double_banana;

% Set up the model
setup;

% Number of particles
N = 1e3;

% Initial particle configuration
x_init = prior.m0 + prior.C0sqrt*randn(model.n,N);  


%% Compare the algorithms after different computational times

itermax = 1e10;

% Set maximum run times
timemax     = [5 10 20];
timemaxdiff = [timemax(1) diff(timemax)];

% Posterior contour
x1 = -2:.05:2; x2 = -2.1:.05:3;
[X1,X2] = meshgrid(x1, x2);

post_pdf  = post2contour([X1(:) X2(:)]', model, prior, obs);
post_pdf  = reshape(post_pdf, length(x2), length(x1));

% Run the algorithms and create plot
figure('name', '2D double banana') 

% SVQN-FI
stepsize  = 1;  
x_SVQN_FI = x_init;

for j = 1:3   
    x_SVQN_FI = SVQN_FI(x_SVQN_FI, stepsize, itermax, timemaxdiff(j), model, prior, obs);

    subplot(4,3,j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_SVQN_FI(1,:), x_SVQN_FI(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVQN-FI -- ' num2str(floor(timemax(j))) 's'])
end

% SVQN-I
stepsize = 1;  
x_SVQN_I = x_init;

for j = 1:3   
    x_SVQN_I = SVQN_I(x_SVQN_I, stepsize, itermax, timemaxdiff(j), model, prior, obs);
    
    subplot(4,3,3 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_SVQN_I(1,:),  x_SVQN_I(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVQN-I -- ' num2str(floor(timemax(j))) 's'])
end

% SVGD-FI
stepsize  = 1e-1;
x_SVGD_FI = x_init;

for j = 1:3   
    x_SVGD_FI = SVGD_FI(x_SVGD_FI, stepsize, itermax, timemaxdiff(j), model, prior, obs);
    
    subplot(4,3,6 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_SVGD_FI(1,:), x_SVGD_FI(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVGD-FI -- ' num2str(floor(timemax(j))) 's'])
end

% SVGD-I
stepsize = 1e-2;
x_SVGD_I = x_init;

for j = 1:3    
    x_SVGD_I = SVGD_I(x_SVGD_I, stepsize, itermax, timemaxdiff(j), model, prior, obs);
    
    subplot(4,3,9 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_SVGD_I(1,:), x_SVGD_I(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVGD-I -- ' num2str(floor(timemax(j))) 's'])  
end


%% Compare the algorithms after different number of iterations

% Set random seed
rng(1);

itermax     = [10 100 1000];
itermaxdiff = [itermax(1) diff(itermax)];

% Set maximum run times
timemax = 1e10;

% Posterior contour
x1 = -2:.05:2; x2 = -2.1:.05:3;
[X1,X2] = meshgrid(x1, x2);

post_pdf  = post2contour([X1(:) X2(:)]', model, prior, obs);
post_pdf  = reshape(post_pdf, length(x2), length(x1));

% Run the algorithms and create plot
figure('name', '2D double banana -- Iterations')

% SVQN-FI
stepsize  = 1;
x_SVQN_FI = x_init; 

for j = 1:3   
    x_SVQN_FI = SVQN_FI(x_SVQN_FI, stepsize, itermaxdiff(j), timemax, model, prior, obs);

    subplot(4,3,j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_SVQN_FI(1,:), x_SVQN_FI(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVQN-FI -- ' num2str(floor(itermax(j))) ' iterations'])
end

% SVQN-I
stepsize = 1;
x_SVQN_I = x_init;

for j = 1:3   
    x_SVQN_I = SVQN_I(x_SVQN_I, stepsize, itermaxdiff(j), timemax, model, prior, obs);
    
    subplot(4,3,3 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_SVQN_I(1,:),  x_SVQN_I(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVQN-I -- ' num2str(floor(itermax(j))) ' iterations'])
end

% SVGD-FI
stepsize  = 1e-1;
x_SVGD_FI = x_init; 

for j = 1:3   
    x_SVGD_FI = SVGD_FI(x_SVGD_FI, stepsize, itermaxdiff(j), timemax, model, prior, obs);
    
    subplot(4,3,6 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_SVGD_FI(1,:), x_SVGD_FI(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVGD-FI -- ' num2str(floor(itermax(j))) ' iterations'])
end

% SVGD-I
stepsize = 1e-2;
x_SVGD_I = x_init;

for j = 1:3    
    x_SVGD_I = SVGD_I(x_SVGD_I, stepsize, itermaxdiff(j), timemax, model, prior, obs);
    
    subplot(4,3,9 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_SVGD_I(1,:), x_SVGD_I(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVGD-I -- ' num2str(floor(itermax(j))) ' iterations'])  
end