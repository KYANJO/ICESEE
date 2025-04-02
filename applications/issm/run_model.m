function run_model(nprocs,k,dt,tinitial,tfinal)
	% function run_model
	
		%  read kwargs from a .mat file
		kwargs 			= load('model_kwargs.mat');
		cluster_name    = char(kwargs.cluster_name);
		steps 			= double(kwargs.steps);
		% dt 				= double(kwargs.dt);
		% tinitial 		= double(kwargs.tinitial);
		% tfinal 			= double(kwargs.tfinal);
		% k = double(kwargs.k);  %time step
		fprintf('[DEBUG-issm] Running the model from time: %f to %f at step %d\n', tinitial, tfinal, k);
	
		%Solving #7
		nprocs = 4;
		if any(steps==7)
			% load the preceding step #help loadmodel
			% path is given by the organizer with the name of the given step
			%->
			md = loadmodel('./Models/ISMIP.BoundaryCondition');
			% Set cluster #md.cluster
			% generic parameters #help generic
			% set only the name and number of process
			%->
			% cluster_name = 'cos2a16204.local'
			md.cluster=generic('name',cluster_name,'np',nprocs);
			% md.cluster=generic('name','cos2a16204.local','np',nprocs );
			% md.cluster=generic('name',oshostname(),'np',nprocs );
			% Set which control message you want to see #help verbose
			%->
			md.verbose=verbose('convergence',true);
			% Solve #help solve
			% we are solving a StressBalanc
	
			%  add time stepping parameters
			md.timestepping.time_step  = dt;
			md.timestepping.start_time = tinitial;
			md.timestepping.final_time = tfinal;
			%->
			md=solve(md,'Stressbalance');
			% save the given model
			%->
			save ./Models/ISMIP.StressBalance md;
			% plot the surface velocities #plotdoc
			%->
			% plotmodel(md,'data',md.results.StressbalanceSolution.Vel)
		end
	
		if any(steps==8)
			% load the preceding step #help loadmodel
			% path is given by the organizer with the name of the given step
			%->
			if k == 0
				% load Boundary conditions from the inital conditions
				md = loadmodel('./Models/ISMIP.BoundaryCondition');
				% time stepping parameters
				md.timestepping.time_step=dt;
				md.timestepping.start_time=tinitial;
				md.timestepping.final_time=tfinal;

				md.cluster=generic('name',cluster_name,'np',nprocs);
				% Set which control message you want to see #help verbose
				%->
				md.verbose=verbose('convergence',true);
				% set the transient model to ignore the thermal model
				% #md.transient
				%->
				md.transient.isthermal=0;

				md=solve(md,'Transient');
				% save the given model
				%->
				save ./Models/ISMIP.Transient md;
			else
				% clear all;	 close all;
				% load the preceding step #help loadmodel
				% md = loadmodel('./Models/ISMIP.Transient');
				% md.initialization.vx 		= md.results.TransientSolution(end).Vx;     
				% md.initialization.vy 		= md.results.TransientSolution(end).Vy;
				% md.initialization.vz 	   	= md.results.TransientSolution(end).Vz;
				% md.initialization.pressure 	= md.results.TransientSolution(end).Pressure;
		
				% % md.geometry.thickness 		= md.results.TransientSolution(end).Thickness;
				% % md.geometry.base 			= md.results.TransientSolution(end).Base;
				% % md.geometry.surface 		= md.results.TransientSolution(end).Surface;
				% % md.smb.mass_balance         = md.results.TransientSolution(end).SmbMassBalance;

				
				% Load previous model
				md = loadmodel('./Models/ISMIP.Transient');

				% Load updates from ensemble_out.mat
				ensemble_out = load('ensemble_out.mat');

				% Update model initial state
				md.initialization.vx       = ensemble_out.Vx;
				md.initialization.vy       = ensemble_out.Vy;
				md.initialization.vz       = ensemble_out.Vz;
				md.initialization.pressure = ensemble_out.Pressure;

				% Time stepping parameters
				md.timestepping.time_step  = dt;
				md.timestepping.start_time = tinitial;
				md.timestepping.final_time = tfinal;

				% Cluster setup
				md.cluster = generic('name', cluster_name, 'np', nprocs);
				md.verbose = verbose('convergence', true);
				md.transient.isthermal = 0;

				% Solve
				md = solve(md, 'Transient');

				% Save model
				save('./Models/ISMIP.Transient', 'md');

				% Save final state to dictionary for Python
				results_out.Vx       = md.results.TransientSolution(end).Vx;
				results_out.Vy       = md.results.TransientSolution(end).Vy;
				results_out.Vz       = md.results.TransientSolution(end).Vz;
				results_out.Pressure = md.results.TransientSolution(end).Pressure;

				% Save just the fields (as flat keys) so Python can read easily
				save('ensemble_out.mat', '-struct', 'results_out');

				% save(sprintf('ensemble_output_%d.mat', rank), '-struct', 'ensemble_out');


			end
	
			% % Set cluster #md.cluster
			% % generic parameters #help generic
			% % set only the name and number of process
			% %->
			% md.cluster=generic('name',cluster_name,'np',nprocs);
			% % Set which control message you want to see #help verbose
			% %->
			% md.verbose=verbose('convergence',true);
			% % set the transient model to ignore the thermal model
			% % #md.transient
			% %->
			% md.transient.isthermal=0;
			% % define the timestepping scheme
			% % everything here should be provided in years #md.timestepping
			% % give the length of the time_step (4 years)
			% %->
			% md.timestepping.time_step=dt;
			% % give final_time (20*4 years time_steps)
			% md.timestepping.start_time=tinitial;
			% md.timestepping.final_time=tfinal;
			% % Solve #help solve
			% % we are solving a TransientSolution
			% %->
			% md=solve(md,'Transient');
			% % save the given model
			% %->
			% save ./Models/ISMIP.Transient md;
			% % plot the surface velocities #plotdoc
			% %->
			% % plotmodel(md,'data',md.results.TransientSolution(20).Vel)
		end
	end