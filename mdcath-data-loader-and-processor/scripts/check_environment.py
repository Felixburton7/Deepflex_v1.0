#!/usr/bin/env python3
"""
Checks the environment for required external tools (DSSP, Aposteriori).
"""
import shutil
import subprocess
import sys
from platform import python_version

# Try to import optional dependencies for extra checks
try:
    import pdbUtils
    PDBUTILS_OK = True
except ImportError:
    PDBUTILS_OK = False

def check_tool(tool_name):
    """Checks if a tool exists in the system PATH."""
    path = shutil.which(tool_name)
    if path:
        print(f"[ OK ] Found '{tool_name}' at: {path}")
        # Optionally, try running with --version if common
        try:
            result = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0 and result.stdout:
                 print(f"       Version info: {result.stdout.strip().splitlines()[0]}") # Show first line
            # Handle cases where version might be in stderr or require different flags
            elif result.returncode == 0 and result.stderr:
                 print(f"       Version info: {result.stderr.strip().splitlines()[0]}")
            # Special case for older DSSP/mkdssp that might just print help or fail
            elif tool_name in ['dssp', 'mkdssp'] and result.returncode != 0:
                 print(f"       Could not get version via --version (normal for some DSSP versions).")
            # elif result.returncode !=0 :
            #      print(f"       '{tool_name} --version' exited with code {result.returncode}")

        except Exception:
             print(f"       Could not determine version for {tool_name}.")
        return True
    else:
        print(f"[FAIL] Command '{tool_name}' not found in PATH.")
        return False

def main():
    print("--- Environment Check for mdcath-processor ---")
    print(f"Python version: {python_version()}")

    print("\nChecking external command-line tools:")
    dssp_found = check_tool("dssp")
    mkdssp_found = check_tool("mkdssp")
    aposteriori_found = check_tool("make-frame-dataset") # Aposteriori's command

    print("\nChecking optional Python libraries:")
    if PDBUTILS_OK:
        print("[ OK ] Found 'pdbUtils' library.")
    else:
        print("[WARN] Optional 'pdbUtils' library not found. PDB cleaning will use fallback.")

    print("\nSummary:")
    final_status_ok = True
    if not (dssp_found or mkdssp_found):
        print("\n[ERROR] DSSP executable ('dssp' or 'mkdssp') is required but not found in PATH.")
        print("        Please install DSSP (e.g., 'conda install -c salilab dssp') and ensure it's accessible.")
        final_status_ok = False

    # Check Aposteriori installation (assuming voxelization is generally enabled)
    # Could add a config check here if needed
    if not aposteriori_found:
        print("\n[ERROR] Aposteriori ('make-frame-dataset' command) is required but not found.")
        print("        Please install Aposteriori ('pip install aposteriori').")
        final_status_ok = False

    if final_status_ok:
         print("\nRequired external tools seem to be available.")
         if not PDBUTILS_OK:
             print("Note: Using fallback PDB cleaning due to missing 'pdbUtils'. Install 'pdbUtils' for potentially better results.")
         print("Environment check passed.")
         sys.exit(0)
    else:
         print("\nEnvironment check failed due to missing required external tools.")
         sys.exit(1)


if __name__ == "__main__":
    main()
