# --- Imports ---
import os
import sys
import subprocess
import numpy as np
import time

CMD_FILE = "matlab_cmd.txt"
LOG_FILE = "matlab_server.log"

class MatlabServer:
    def __init__(self, issm_dir):
        self.cmd_file = CMD_FILE
        self.log_file = LOG_FILE
        self.process = None
        self.issm_dir = issm_dir

    def start(self):
        if os.path.exists(self.cmd_file):
            os.remove(self.cmd_file)

        matlab_cmd = (
            "matlab -nodisplay -nosplash -nodesktop "
            f"-r \"addpath(genpath('{self.issm_dir}')); run('issm_env'); "
            f"matlab2python/matlab_server('{self.cmd_file}');\""
        )

        print("[MATLAB Server] Launching MATLAB in background...")
        self.process = subprocess.Popen(
            matlab_cmd,
            shell=True,
            stdout=open(self.log_file, 'w'),
            stderr=subprocess.STDOUT
        )
        time.sleep(3)
        print("[MATLAB Server] Launched.")

    def send_command(self, command):
        print(f"[MATLAB Server] Sending command: {command}")
        with open(self.cmd_file, 'w') as f:
            f.write(command)

        while os.path.exists(self.cmd_file):
            time.sleep(0.5)

    def shutdown(self):
        print("[MATLAB Server] Shutting down...")
        self.send_command("exit")
        self.process.wait()
        print("[MATLAB Server] Shutdown complete.")

# # Example usage:
# if __name__ == '__main__':
#     ISSM_DIR = os.environ.get("ISSM_DIR")
#     server = MatlabServer(ISSM_DIR)
#     server.start()
#     server.send_command("run_model(4, 0, 0.5, 0.0, 4.0)")
#     server.shutdown()



def subprocess_cmd_run(issm_cmd, nprocs: int, verbose: bool = True):
    """
    Run ISSM using a MATLAB script via subprocess.Popen.

    Parameters:
    - issm_cmd: Full command string to run ISSM in MATLAB
    - nprocs: Number of processors to pass to runme.m (for display/debug)
    - verbose: If True, print stdout (trimmed) and stderr (only if non-empty)
    """
    try:
        process = subprocess.Popen(
            issm_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        stdout, stderr = process.communicate()

        if verbose:
            stdout_lines = stdout.splitlines()
            trimmed_stdout = "\n".join(stdout_lines[9:])  # Skip banner
            print(f"\n[ICESEE] ➤ Running ISSM with {nprocs} processors")
            print("------ ICESEE<->MATLAB STDOUT ------")
            print(trimmed_stdout.strip())

            if stderr.strip():  # Only print stderr if there's content
                print("------ ICESEE<->MATLAB STDERR ------")
                print(stderr.strip())

        if process.returncode != 0:
            # raise subprocess.CalledProcessError(
            #     process.returncode, issm_cmd, output=stdout, stderr=stderr
            # )
            raise subprocess.CalledProcessError(process.returncode, issm_cmd)

    except FileNotFoundError:
        print("❌ Error: MATLAB not found in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"❌ MATLAB exited with error code {e.returncode}")
        if e.stderr.strip():
            print("------ MATLAB STDERR ------")
            print(e.stderr.strip())
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        

# --- MATLAB Engine Initialization ---
# MATLAB Engine Initialization
def initialize_matlab_engine():
    """
    Initializes the MATLAB Engine for Python.

    Returns:
        eng: The MATLAB Engine instance.
    """
    try:
        import matlab.engine
        print("Starting MATLAB engine...")
        # Start a headless MATLAB engine without GUI
        eng = matlab.engine.start_matlab("-nodisplay -nosplash -nodesktop -nojvm")
        print("MATLAB engine started successfully.")
        return eng
    except ImportError:
        print("MATLAB Engine API for Python not found. Attempting to install...")

        try:
            # Find MATLAB root
            matlab_root = find_matlab_root()

            # Install the MATLAB Engine API for Python
            install_matlab_engine(matlab_root)

            # Retry importing and starting the MATLAB Engine
            import matlab.engine
            eng = matlab.engine.start_matlab("-nodisplay -nosplash -nodesktop -nojvm")
            print("MATLAB engine started successfully after installation.")
            return eng
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MATLAB Engine API for Python: {e}\n"
                "Ensure MATLAB is installed, and its bin directory is added to your PATH.\n"
                "For instructions on installing the MATLAB Engine API for Python, see "
                "the official MATLAB documentation or the provided README.md."
            )

# # --- MATLAB Root Finder ---
def find_matlab_root():
    """
    Finds the MATLAB root directory by invoking MATLAB from the terminal.

    Returns:
        matlab_root (str): The root directory of the MATLAB installation.
    """
    try:
        # Run MATLAB in terminal mode to get matlabroot
        result = subprocess.run(
            ["matlab", "-batch", "disp(matlabroot)"],  # Run MATLAB with -batch mode
            text=True,
            capture_output=True,
            check=True
        )
        
        # Extract and clean the output
        matlab_root = result.stdout.strip()
        print(f"MATLAB root directory: {matlab_root}")
        return matlab_root
    except FileNotFoundError:
        print(
            "MATLAB is not available in the system's PATH. "
            "Ensure MATLAB is installed and its bin directory is in the PATH."
        )
    except subprocess.CalledProcessError as e:
        print(f"Error while executing MATLAB: {e.stderr.strip()}")
        raise

# --- MATLAB Engine Installation ---
def install_matlab_engine(matlab_root):
    """
    Installs the MATLAB Engine API for Python using the MATLAB root directory.

    Args:
        matlab_root (str): The root directory of the MATLAB installation.
    """
    try:
        # Save the current working directory
        current_dir = os.getcwd()

        # Path to the setup.py script for MATLAB Engine API for Python
        setup_path = os.path.join(matlab_root, "extern", "engines", "python")
        assert os.path.exists(setup_path), f"Setup path does not exist: {setup_path}"

        # Change to the setup directory
        os.chdir(setup_path)

        # Run the setup.py script to install the MATLAB Engine API
        print("Installing MATLAB Engine API for Python...")
        result = subprocess.run(
            ["python", "setup.py", "install", "--user"],
            text=True,
            capture_output=True,
            check=True
        )

        # Export the build directory to PYTHONPATH
        home_path = os.path.expanduser("~/")  # Adjust if needed
        os.environ["PYTHONPATH"] = f"{home_path}/lib:{os.environ.get('PYTHONPATH', '')}"

        print("MATLAB Engine API for Python installed successfully.")
    except AssertionError as e:
        print(f"AssertionError: {e}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error while installing MATLAB Engine API for Python:\n{e.stderr.strip()}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    finally:
        # Change back to the original directory
        os.chdir(current_dir)

# Example usage
if __name__ == "__main__":
    matlab_root = find_matlab_root()
    # install_matlab_engine(matlab_root)
