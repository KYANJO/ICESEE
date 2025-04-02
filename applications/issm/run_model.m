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
		rank = 0;
		filename = sprintf('ensemble_output_%d.h5', rank);
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
			% Construct filename based on rank
			filename = sprintf('ensemble_output_%d.h5', rank);
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

				% save these fields to a file for ensemble use
				fields = {'Vx', 'Vy', 'Vz', 'Pressure'};
				result = md.results.TransientSolution(end);
				save_ensemble_hdf5(filename, result, fields);
			else
				
				% Load previous model
				md = loadmodel('./Models/ISMIP.Transient');
				filename = sprintf('ensemble_output_%d.h5', rank);
				% load from an h5 file
				md.initialization.vx       = h5read(filename, '/Vx');
				md.initialization.vy       = h5read(filename, '/Vy');
				md.initialization.vz       = h5read(filename, '/Vz');
				md.initialization.pressure = h5read(filename, '/Pressure');

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

				% save these fields to a file for ensemble use
				fields = {'Vx', 'Vy', 'Vz', 'Pressure'};
				result = md.results.TransientSolution(end);
				save_ensemble_hdf5(filename, result, fields);

			end
		end
	end

	function save_ensemble_hdf5(filename, result, field_names)
		
		% Remove file if it already exists
		if isfile(filename)
			delete(filename);
		end
	
		% Iterate over each requested field
		for i = 1:length(field_names)
			field = field_names{i};
	
			% Check field exists in result
			if isfield(result, field)
				data = result.(field);
				h5create(filename, ['/' field], size(data));
				h5write(filename, ['/' field], data);
			else
				warning('Field "%s" not found in result. Skipping.', field);
			end
		end
	
		fprintf('[HDF5] Saved: %s\n', filename);
	end