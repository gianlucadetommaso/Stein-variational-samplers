%% Script double banana test case
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
load_dir_double_banana;

% Set up the model
setup;

% Number of particles
N = 1e3;

% Initial particle configuration
x_init = prior.m0 + prior.C0sqrt*randn(model.n,N);  

% Number of iterations
itermax = 100;

% Estimate computational time
stepsize = 1;
[~, ~, t_NH]  = SVN_H(x_init, stepsize, itermax, model, prior, obs);

stepsize = 1;
[~, ~, t_NI]  = SVN_I(x_init, stepsize, itermax, model, prior, obs);

stepsize = 1e-1;
[~, ~, t_GDH] = SVGD_H(x_init, stepsize, itermax, model, prior, obs);

stepsize = 1e-1;
[~, ~, t_GDI] = SVGD_I(x_init, stepsize, itermax, model, prior, obs);

% Time ratios with respect to SVN-H
r_NI  = t_NH / t_NI;
r_GDH = t_NH / t_GDH;
r_GDI = t_NH / t_GDI;

% Posterior contour
x1 = -2:.05:2;
x2 = -2.1:.05:3;
[X1,X2] = meshgrid(x1, x2);

post_pdf = post2contour([X1(:) X2(:)]', model, prior, obs);
post_pdf = reshape(post_pdf, length(x2), length(x1));

% Figure
figure('name', '2D double banana')


%% Compare the algorithms with same total cost

% SVN-H
itermax_NH     = [10 50 100];
itermaxdiff_NH = [itermax_NH(1) diff(itermax_NH)];
stepsize_NH    = 1;  
x_NH           = x_init;

for j = 1:3   
    % Run SVN-H
    [x_NH, stepsize_NH] = SVN_H(x_NH, stepsize_NH, itermaxdiff_NH(j), model, prior, obs);

    % Plot
    subplot(4,3,j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_NH(1,:), x_NH(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
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
    
    % Plot
    subplot(4,3,3 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_NI(1,:),  x_NI(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
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
    
    % Plot 
    subplot(4,3,6 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_GDH(1,:), x_GDH(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVGD-H -- ' num2str(floor(itermax_GDH(j))) ' iterations'])
end

% SVGD-I
itermax_GDI     = ceil(r_GDI*itermax_NH); 
itermaxdiff_GDI = [itermax_GDI(1) diff(itermax_GDI)];
stepsize_GDI    = 5*1e-2;
x_GDI           = x_init;

for j = 1:3  
    % Run SVGD-I
    [x_GDI, stepsize_GDI] = SVGD_I(x_GDI, stepsize_GDI, itermaxdiff_GDI(j), model, prior, obs);
    
    % Plot 
    subplot(4,3,9 + j)
    hold on
    contourf(X1, X2, post_pdf)
    plot(x_GDI(1,:), x_GDI(2,:), 'g.', 'markersize', 5)
    xlim([-2 2]), ylim([-2 3])
    title(['SVGD-I -- ' num2str(floor(itermax_GDI(j))) ' iterations'])
end
