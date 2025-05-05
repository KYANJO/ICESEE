# Steps to setup Matlab from python api
The following steps ensure that the MATLAB engine for Python is correctly set up on macOS or Linux systems

## On a local machine with sudo previlleges
1. **Ensure MATLAB is Installed:**
   - Make sure that MATLAB is installed and properly licensed on your system.

2. **Identify the MATLAB Root Directory:**
   - Open MATLAB and type `matlabroot` in the command window. This command will display the root directory of your MATLAB installation.

3. **Navigate to the MATLAB Engine Directory:**
   - Open a terminal and change to the MATLAB engine directory:
     ```bash
     cd <MATLAB_ROOT_DIRECTORY>/extern/engines/python
     ```
     Replace `<MATLAB_ROOT_DIRECTORY>` with the output of `matlabroot`.

4. **Install the MATLAB Engine for Python:**
   - Run the following command to install the engine using Python 3:
     ```bash
     sudo python3 setup.py install
     ```
     - Ensure the Python version is compatible (3.9, 3.10, or 3.11).

5. **Add the Build Directory to the PYTHONPATH:**
   - Export the build directory path to the `PYTHONPATH` environment variable:
     ```bash
     export PYTHONPATH=<MATLAB_ROOT_DIRECTORY>/extern/engines/python/build/lib:$PYTHONPATH
     ```
     Replace `<MATLAB_ROOT_DIRECTORY>` with the MATLAB root directory.

6. **Verify Installation:**
   - Open a Python shell and try importing the MATLAB engine to confirm the installation:
     ```python
     import matlab.engine
     ```

---

### Steps for Non-Sudo Installation of MATLAB Engine:
If you are installing the MATLAB engine for Python on a cluster or any environment where you lack `sudo` privileges, you can follow these alternative steps to set it up locally within your user environment:

1. **Ensure MATLAB is Installed:**
   - Verify that MATLAB is installed on the cluster and accessible via your user account.

2. **Identify the MATLAB Root Directory:**
   - Log in to the cluster, start MATLAB, and run:
     ```matlab
     matlabroot
     ```
   - Note the output (e.g., `/path/to/matlab`).

3. **Navigate to the MATLAB Engine Directory:**
   - In the cluster shell, navigate to the engine installation directory:
     ```bash
     cd /path/to/matlab/extern/engines/python
     ```
     Replace `/path/to/matlab` with the actual MATLAB root directory.

4. **Install the MATLAB Engine Locally for Your User Account:**
   - Use the `--user` flag with the `setup.py` installation command to install it in your home directory:
     ```bash
     python3 setup.py install --user
     ```

5. **Add the Build Directory to Your PYTHONPATH:**
   - Find the build directory path after installation. It should be something like:
     ```
     /path/to/matlab/extern/engines/python/build/lib
     ```
   - Add it to your `PYTHONPATH` environment variable in your shell profile file (e.g., `~/.bashrc` or `~/.bash_profile`):
     ```bash
     export PYTHONPATH=/path/to/matlab/extern/engines/python/build/lib:$PYTHONPATH
     ```
   - Reload your shell profile:
     ```bash
     source ~/.bashrc
     ```

6. **Verify Installation:**
   - Open a Python shell and try importing the MATLAB engine:
     ```python
     import matlab.engine
     ```

7. **Set Environment Variables for MATLAB if Needed:**
   - On some clusters, you may need to set additional environment variables to ensure MATLAB can run properly. For example:
     ```bash
     export LD_LIBRARY_PATH=/path/to/matlab/bin/glnxa64:$LD_LIBRARY_PATH
     export PATH=/path/to/matlab/bin:$PATH
     ```

---

### Additional Notes:
- If `setup.py` fails due to directory write permissions, you can manually build the library using:
  ```bash
  python3 setup.py build --build-base=/your/home/directory/build_dir
  ```
  Then, update `PYTHONPATH` to point to this `build/lib` directory.

- For shared environments, consider using a virtual environment or `conda` environment to isolate the installation:
  ```bash
  python3 -m venv ~/my_matlab_env
  source ~/my_matlab_env/bin/activate
  python3 setup.py install
  ```
