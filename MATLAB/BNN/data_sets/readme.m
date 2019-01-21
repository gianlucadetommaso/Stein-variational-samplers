% description of all data sets
%
% All data sets are downloaded from 
%   https://archive.ics.uci.edu/ml/machine-learning-databases/
%
% For all data sets, the the prediction variables (y) are placed at right 
% columns, and other left columns are input variables
%
% Tiangang Cui, 3 August, 2018
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       data name               data size   inputs  outputs 
%
% 1.    boston_housing          506         13      1       continous
% 2.    concrete_stength        1030        8       1       continous
% 3.    energy_efficiency       768         8       2       continous
% 4.    kin8nm                  8192        8       1       continous
% 5.    naval_propulsion        11934       16      2       continous
% 6.    power_plant             9568        4       1       continous
% 7.    protein_structure       45730       9       1       continous
% 8.    wine_quality_r          1599        11      1       integer
% 9.    wine_quality_w          4898        11      1       integer
% 10.   yacht_hydro             308         6       1       continous
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_names = {  'boston_housing',   13, 1; ...
                'concrete_strength', 8, 1; ...
                'energy_efficiency', 8, 2; ...
                'kin8nm',            8, 1; ...
                'naval_propulsion', 16, 2; ...
                'power_plant',       4, 1; ...
                'protein_structure', 9, 1; ...
                'wine_quality_r',   11, 1; ...
                'wine_quality_w',   11, 1; ...
                'yacht_hydro',       6, 1};
            
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The last data set is the year prediction data set. The goal is to predict
% the year of the song by other 90 variables. 
% 
% This large data is splitted into a training set (463,715 entries) and 
% a test set (51,630 entries)
%
% the prediction variable is integer valued
%