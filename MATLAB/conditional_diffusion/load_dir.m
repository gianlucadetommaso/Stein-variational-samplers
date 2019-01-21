clear all
close all

% The root directory, used in the problem definition script for saving the
% data. 

[~,tmp] = fileattrib('../');
root    = tmp.Name;

% remove current path
addpath(genpath(root));
rmpath(genpath(root));

% add working directory from fastfins
root    = [root, '/fastfins'];
addpath(root);
addpath(genpath([root '/solvers']));
addpath(genpath([root '/library']));
addpath(genpath([root '/optimizer']));
addpath(genpath([root '/samplers']));

% add local working directory
addpath([pwd '/cd_model']);

% set default for figures, you may not need this
figure_default;
