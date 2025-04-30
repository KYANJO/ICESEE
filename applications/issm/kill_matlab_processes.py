import psutil
import platform
import signal

def kill_matlab_processes():
    matlab_count = 0
    for proc in psutil.process_iter(['name', 'pid', 'cmdline']):
        try:
            # Check for MATLAB processes (name varies by OS)
            if 'matlab' in proc.info['name'].lower() or 'MATLAB' in proc.info['name']:
                print(f"Found MATLAB process: {proc.info['name']} (PID: {proc.info['pid']})")
                
                # Get command-line arguments to check for GUI-related flags
                cmdline = proc.info['cmdline']
                is_gui = True  # Assume GUI unless proven otherwise
                
                # Check command-line arguments for non-GUI flags
                if cmdline and any(flag in cmdline for flag in ['-nodisplay', '-nodesktop']):
                    is_gui = False  # Non-GUI instance
                
                if not is_gui:
                    # Terminate non-GUI process
                    if platform.system() == "Windows":
                        proc.terminate()  # Windows uses terminate
                    else:
                        proc.send_signal(signal.SIGTERM)  # Unix uses SIGTERM
                    matlab_count += 1
                    print(f"Terminated MATLAB process (PID: {proc.info['pid']})")
                else:
                    print(f"Skipped GUI MATLAB process (PID: {proc.info['pid']})")
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    if matlab_count == 0:
        print("No non-GUI MATLAB processes found to terminate.")
    else:
        print(f"Terminated {matlab_count} non-GUI MATLAB process(es).")

kill_matlab_processes()





