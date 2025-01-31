% =============================================================================
% Author: Brian Kyanjo
% Date: 2025-01-28
% Description: 
% This script implements the ISSM model with data assimilation using the 
% Ensemble Kalman Filter (EnKF). It leverages helper functions and classes 
% from the Python script `python2matlab/_call_da_issm.py`.
%
% Workflow:
% - During the forecast stage, the model is run Nens times (number of ensembles) 
%   to generate the ensemble forecast.
% - The forecast is analyzed using the EnKF to produce the analysis.
% - The analysis is then used as the initial condition for the subsequent forecast.
%
% The process is repeated for a specified number of cycles. Between cycles, 
% observations and model states are updated. A restart file is used to:
% - Save the model state at a specific time step.
% - Perform the forecast and analysis stages.
%
% =============================================================================

% --- specify the path to the python environment ---
pyenv('Version','/Users/bkyanjo3/firedrake/bin/python');

% Make the $ISSM_DIR environment variable available
issm_dir = getenv('ISSM_DIR');  % Retrieve the ISSM_DIR environment variable

if ~isempty(issm_dir)
    % Check if the directory exists
    if isfolder(issm_dir)
        % Add the ISSM directory and all its subdirectories to the MATLAB path
        addpath(genpath(issm_dir));
        disp(['Added ISSM directory from path: ', issm_dir]);
    else
        error('The ISSM_DIR directory does not exist: %s', issm_dir);
    end
else
    error('ISSM_DIR is not set. Please set the ISSM_DIR environment variable.');
end

% Make the examples directory available
examples_dir = fullfile(issm_dir, 'examples', 'SquareIceShelf');  % Path to the examples directory

if isfolder(examples_dir)
    % Add the examples directory to the MATLAB path
    addpath(examples_dir);
    disp(['Added examples directory from path: ', examples_dir]);
else
    error('The examples directory does not exist: %s', examples_dir);
end

% List the contents of the examples directory
% examples_contents = dir(examples_dir);
% disp('Contents of the examples directory:');
% for i = 1:length(examples_contents)
%     if ~examples_contents(i).isdir
%         fprintf('File: %s\n', examples_contents(i).name);
%     else
%         fprintf('Directory: %s\n', examples_contents(i).name);
%     end
% end

% get the current working directory
original_dir = pwd;

% change to the examples directory
cd(examples_dir);
% Debug: Check the current directory
disp(['Current working directory: ', pwd]);

% run the script
run('runme.m');

% plot the results
plotmodel(md,'data',md.results.StressbalanceSolution.Vel);

% go back to the original directory to call the python script
cd(original_dir);
disp(['Current working directory: ', pwd]);

