# --- Imports ---
import os
import sys
import subprocess
import numpy as np

# --- MATLAB Engine Initialization ---
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
        eng = matlab.engine.start_matlab("-nodisplay -nosplash -nodesktop")
        print("MATLAB engine started successfully.")
        return eng
    except ImportError as e:
        try:
            # find matlabroot
            matlabroot = os.environ.get("MATLABROOT")
            
        # assert False, (
        #     "MATLAB Engine API for Python not installed.\n"
        #     "Please follow the instructions in matlab2python/README.md to install it, "
        #     "assuming MATLAB is already installed on your system.\n"
        # )
        except ImportError as e:
            print("MATLAB Engine API for Python not installed.\n"
                  "Please follow the instructions in matlab2python/README.md to install it, "
                  "assuming MATLAB is already installed on your system.\n")
            sys.exit(1)

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

if __name__ == "__main__":
    find_matlab_root()