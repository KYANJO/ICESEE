issm_dir = getenv('ISSM_DIR');
if ~isempty(issm_dir)
    addpath(genpath(issm_dir));
    disp(['Added ISSM directory and subdirectories from path: ', issm_dir]);
else
    error('ISSM_DIR is not set.');
end