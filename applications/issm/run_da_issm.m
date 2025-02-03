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

clear all; close all; clc; 

% --- Get the current script directory ---
script_dir = fileparts(mfilename('fullpath'));
cd(script_dir); % Change to the script directory
disp(['Current working directory: ', pwd]);

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
% original_dir = pwd;

% change to the examples directory
cd(examples_dir);
% Debug: Check the current directory
disp(['Current working directory: ', pwd]);

% Number of ensembles
Nens = 10;

md=model; % create an empty model structure
md=triangle(md,'DomainOutline.exp',50000); % create a mesh of the domain of 50km
md=setmask(md,'all',''); % define glacier systme as an ice shelf
md=parameterize(md,'Square.par'); % parameterize the model with square.par
md=setflowequation(md,'SSA','all'); % define all elements as SSA elements
md.cluster=generic('name',oshostname,'np',2); % set the cluster

% md = solve(md,'Stressbalance'); % compute the velocity filed of the ice shelf

% ensemble parameters
num_state_variables = 3; % [h,u,smb]
hdim = size(md.geometry.thickness);

% vdim = size(md.results.StressbalanceSolution.Vel);

% intialize the ensemble ( we will perturb the initial state)
disp('Initialize ensemble');
% ensemble = zeros(num_state_variables*hdim(1),Nens);
ensemble = zeros(hdim(1),Nens);
[ensemble,md_] = initialize_ensemble(md,ensemble,0,0,0);

% plot the results
% md.results.StressbalanceSolution.Vel = ensemble(:,1);
% plotmodel(md,'data',md.results.StressbalanceSolution.Vel);

% call the forecast step
disp('Forecast step');
% [ensemble, md_ensemble] = forecast_step(md,ensemble,0.1,0.1,2);


% timestep = 0.1; % 1 year
% starttime = 0; % 0 year
% finaltime = 2; % 1 year

% % forecast step
% % for each ensemble member, run ISSM to generate the ensemble forecast.
% %  we will then use the ISSM restart functionality to load the model state
% %  at the begining of the timestep and save the model state at the end of the
% %  timestep. 
% for i = 1:Nens
%     % Load the ensemble state
%     filename = ['ensemble_',num2str(i),'.mat'];
%     load(filename,'md_ensemble');
%     md = md_ensemble{i};

%     % set time step
%     md.timestepping.time_step = timestep; % 1 year
%     md.timestepping.start_time = starttime; % 0 year
%     md.timestepping.final_time = finaltime; % 1 year


%     % solve for one timestep
%     md = solve(md,'Stressbalance');
%     % md = solve(md,'Transient');

%     % Run ISSM to generate the ensemble forecast
%     % save the model state at the end of the timestep
%     % save the model state as a restart file for ISSM to read
%     filename = ['forecast_',num2str(i),'.mat'];
%     save(filename,'md');
% end

% analysis step
% perform the EnKF analysis step using the ensemble forecast and observations
%  to update the model state.


% go back to the original directory to call the python script
% cd(original_dir);
cd(script_dir);
disp(['Current working directory: ', pwd]);

% initialize the ensemble
function [ensemble,md] = initialize_ensemble(md,ensemble, timestep, starttime, finaltime)

    md.timestepping.time_step = timestep; % 1 year
    md.timestepping.start_time = starttime; % 0 year
    md.timestepping.final_time = finaltime; % 1 year

    % extract ensemble size
    % ens_size = size(ensemble);
    hdim = size(md.geometry.thickness);
    % hdim = ens_size(1)//num_state_variables;

    % perturb the initial state
    perturbations = 0.05 * randn(hdim); % 5% perturbation
    % md.geometry.thickness = md.geometry.thickness + perturbations;
    md.smb.mass_balance = md.smb.mass_balance + perturbations;

    % Load the initial model state
    md=solve(md,'Stressbalance');

    % define ensemble array
    for ens=1:size(ensemble,2)
         %  Load initial model state
        md_ensemble{ens} = md;

        % save array of velocity fields
        ensemble(:,ens) = md_ensemble{ens}.results.StressbalanceSolution.Vel;
    end
end


% forecaste step
function [ensemble, md_ensemble]=forecast_step(md,ensemble, timestep, starttime, finaltime)
    
    % set time step
    md.timestepping.time_step = timestep; % 1 year
    md.timestepping.start_time = starttime; % 0 year
    md.timestepping.final_time = finaltime; % 1 year

    for ens=1:size(ensemble,2)
        % Load the ensemble state
        md_ensemble{ens} = md;
        
        % set the thickness
        md_ensemble{ens}.geometry.thickness = ensemble(:,ens);
        % solve for the the given timestep
        md_ensemble{ens} =solve(md_ensemble{ens},'Stressbalance');

        % update the ensemble
        ensemble(:,ens) = md_ensemble{ens}.geometry.thickness;
    end

end