%% Script double banana test case
%
% By Gianluca Detommaso 18/05/2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Set random seed
rng(1);

% Load directories
load_dir_double_banana;

% Set up the model
setup;

% Number of particles
N = 1e2;

% Initial particle configuration
x_init = prior.m0 + prior.C0sqrt*randn(model.n,N);  

% Number of iterations
itermax = 20;

% Estimate computational time
stepsize = 1;
xfull = SVNfull_H(x_init, stepsize, itermax, model, prior, obs);
xbd = SVN_H(x_init, stepsize, itermax, model, prior, obs);
xcg = SVNCG_H(x_init, stepsize, itermax, model, prior, obs);

% Posterior contour
x1 = -2:.05:2;
x2 = -2:.05:2;
[X1,X2] = meshgrid(x1, x2);

post_pdf = post2contour([X1(:) X2(:)]', model, prior, obs);
post_pdf = reshape(post_pdf, length(x2), length(x1));

close all, figure %#ok<DUALC>
subplot(2,2,1), contourf(X1, X2, post_pdf)
xlim([-2 2]), ylim([-2 2])
legend('target')
subplot(2,2,2), contourf(X1, X2, post_pdf), hold on
f1 = plot(xfull(1,:), xfull(2,:), 'g.', 'markersize', 10);
xlim([-2 2]), ylim([-2 2])
legend(f1, 'SVNfull-H')
subplot(2,2,3), contourf(X1, X2, post_pdf), hold on
f2 = plot(xbd(1,:), xbd(2,:), 'g.', 'markersize', 10);
xlim([-2 2]), ylim([-2 2])
legend(f2, 'SVNbd-H')
subplot(2,2,4), contourf(X1, X2, post_pdf), hold on
f3 = plot(xcg(1,:), xcg(2,:), 'g.', 'markersize', 10);
xlim([-2 2]), ylim([-2 2])
legend(f3, 'SVNCG-H')