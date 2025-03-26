function run_model(nprocs)
    % Steps
    steps = [7];

    %Solving #7
    if any(steps==7)
	    % load the preceding step #help loadmodel
	    % path is given by the organizer with the name of the given step
	    %->
	    md = loadmodel('./Models/ISMIP.BoundaryCondition');
	    % Set cluster #md.cluster
	    % generic parameters #help generic
	    % set only the name and number of process
	    %->
	    md.cluster=generic('name',oshostname(),'np',nprocs);
	    % Set which control message you want to see #help verbose
	    %->
	    md.verbose=verbose('convergence',true);
	    % Solve #help solve
	    % we are solving a StressBalanc
	    %->
	    md=solve(md,'Stressbalance');
	    % save the given model
	    %->
	    save ./Models/ISMIP.StressBalance md;
	    % plot the surface velocities #plotdoc
	    %->
	    % plotmodel(md,'data',md.results.StressbalanceSolution.Vel)
    end
end