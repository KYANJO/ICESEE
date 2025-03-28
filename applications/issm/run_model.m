function run_model(nprocs)
% function run_model

    %  read kwargs from a .mat file
	kwargs 			= load('model_kwargs.mat');
    cluster_name = char(kwargs.cluster_name)
	steps 			= double(kwargs.steps);
	dt 				= double(kwargs.dt);
	tinitial 		= double(kwargs.tinitial);
	tfinal 			= double(kwargs.tfinal);

	% Check type and convert properly
	% if iscell(kwargs.cluster_name)
	% 	cluster_name = char(kwargs.cluster_name{1});  % Extract and convert from cell
	% else
	% 	cluster_name = char(kwargs.cluster_name);  % Convert if it's a string or array
	% end

    %Solving #7
	% cluster_name = char(cluster_name);
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
	    md = loadmodel('./Models/ISMIP.BoundaryCondition');
	    % Set cluster #md.cluster
	    % generic parameters #help generic
	    % set only the name and number of process
	    %->
	    md.cluster=generic('name',oshostname(),'np',2);
	    % Set which control message you want to see #help verbose
	    %->
	    md.verbose=verbose('convergence',true);
	    % set the transient model to ignore the thermal model
	    % #md.transient
	    %->
	    md.transient.isthermal=0;
	    % define the timestepping scheme
	    % everything here should be provided in years #md.timestepping
	    % give the length of the time_step (4 years)
	    %->
	    md.timestepping.time_step=dt;
	    % give final_time (20*4 years time_steps)
	    md.timestepping.start_time=tinitial;
	    md.timestepping.final_time=tfinal;
	    % Solve #help solve
	    % we are solving a TransientSolution
	    %->
	    md=solve(md,'Transient');
	    % save the given model
	    %->
	    save ./Models/ISMIP.Transient md;
	    % plot the surface velocities #plotdoc
	    %->
	    plotmodel(md,'data',md.results.TransientSolution(20).Vel)
    end
end