clear all
close all

% The root directory, used in the problem definition script for saving the
% data. 

root    = '';   % setup root directry for fastfins

% remove current path
addpath(genpath(root));
rmpath(genpath(root));

% add working directory
addpath(root);
addpath(genpath([root '/solvers']));
addpath(genpath([root '/library']));
addpath(genpath([root '/optimizer']));
addpath(genpath([root '/samplers']));

% add working directory, additional modules
addpath(genpath([root '/mesh_2d'])); % load the mesh, compulsary for using system PDE and prior
addpath(genpath([root '/prior']));   % load the system prior
addpath(genpath([root '/PDEs']));    % load the system PDEs


% set default for figures, you may not need this
figure_default;
