import subprocess
import os
import time
import numpy as np
from typing import Optional, Tuple

def run_fast(fast_exe_path: str, dat_file_name_base: str, timeout: int = 200 
             ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Runs the FAST program, reads the output, and extracts the matrix [CM] and the stress matrix [SIGMA].
    
    Args:
        fast_exe_path (str): Full path to the FAST executable.
        dat_file_name_base (str): Input file name .dat, without the extension.
        
    Returns:
        tuple: (np.ndarray, np.ndarray) The extracted [CM] matrix and [SIGMA] stress matrix.
        
    Raises:
        RuntimeError: If there are problems running or reading the file.
    """
    working_dir = os.path.dirname(fast_exe_path)
    print(f"Running FAST on: {fast_exe_path}")
    print(f"With input file: {dat_file_name_base}")
    print(f"In the working directory: {working_dir}")

    try:
        # Starts process, redirecting stdin (Standart Input) and stdout (Standart Output)
        proc = subprocess.Popen(
            [fast_exe_path],
            cwd=working_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send the file name (without .dat) + enter
        stdout, stderr = proc.communicate(dat_file_name_base + "\n", timeout=timeout)

        print("\n=========== FAST OUTPUT ===========\n")
        print(stdout)
        print("\n=========== END OF OUTPUT ============\n")

        # Split by lines
        lines = stdout.splitlines()

        # --- Extract [CM] matrix: after a line starting with [CM] ---
        cm_matrix = None
        sigma_vec = None

        # Search for line containing [CM]
        idx_cm = None
        for i, line in enumerate(lines):
            if line.strip().startswith("[CM]"):
                idx_cm = i
                break
        if idx_cm is None:
            raise RuntimeError("Marker '[CM]' not found in either FAST output.")

        # Collect stresses from subsequent lines until we have 9 (3x3) or 36 (6x6) floats
        floats = []
        for line in lines[idx_cm+1:]:
            if not line.strip():
                # stop at blank line if already have [CM]
                if floats:
                    break
                else:
                    continue
            line = line.replace(',', '.').split()
            for l in line:
                try:
                    v = float(l)
                    floats.append(v)
                except:
                    # skip not floats
                    pass
                if len(floats) >= 36:
                    break

        if len(floats) >= 36:
            cm_matrix = np.array(floats[:36]).reshape(6,6)
        elif len(floats) >= 9:
            cm_matrix = np.array(floats[:9]).reshape(3,3)
        else:
            raise RuntimeError(f"Incomplete [CM] matrix in FAST output.")
            
        print("\nMatrix [CM] extracted successfully:")
        
        # Extract {AvrStr} average stress line:
        for line in lines:
            if "{AvrStr}" in line:
                stresses = line.replace('{AvrStr}', ' ').replace(',', '.').split()
                numbers = []
                for s in stresses:
                    try:
                        numbers.append(float(s))
                    except:
                        pass
                idx = lines.index(line)
                j = idx + 1
                while len(numbers) < 6 and j < len(lines):
                    values = lines[j].replace(',', '.').split()
                    for v in values:
                        try:
                            numbers.append(float(v))
                        except:
                            pass
                    j += 1
                if numbers:
                    sigma_vec = np.array(numbers)
                break

        print("\nAverage Stresses extracted successfully:")
        print(sigma_vec)

        return cm_matrix, sigma_vec

    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("FAST execution timeout.")
    except FileNotFoundError:
        raise RuntimeError(f"FAST executable not found: '{fast_exe_path}'")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

def CalcPropPlaneStress(C):
    """
    Calculates properties for Plane Stress.
    
    Args:
        C (np.ndarray): Stiffness matrix.
        
    Returns:
        tuple: (E, nu, G1, G2)
    """
    nu = C[0, 1] / C[0, 0]
    E = C[0, 0] * (1 - nu**2)
    G1 = C[2, 2]
    G2 = E / (2 * (1 + nu)) 
    return E, nu, G1, G2

def CalcPropPlaneStrain(C):
    """
    Calculates properties for Plane Strain.
    
    Args:
        C (np.ndarray): Stiffness matrix.
        
    Returns:
        tuple: (E, nu, G1, G2)
    """
    r = C[0, 1] / C[0, 0]
    nu = r / (1 + r)
    E = C[0, 0] * (1 + nu) * (1 - 2 * nu) / (1 - nu)
    G1 = C[2, 2]
    G2 = E / (2 * (1 + nu)) 
    return E, nu, G1, G2

def CalcPropPlaneStrain3x3(C):
    """
    Calculates properties for Plane Strain (for case 3x3).
    
    Args:
        C (np.ndarray): Stiffness matrix.
        
    Returns:
        tuple: (E11, E22, G12, nu12, nu21)
    """
    S = np.linalg.inv(C)
    E22 = 1 / S[0, 0] 
    E33 = 1 / S[1, 1] 
    G23 = 1 / S[2, 2] 
    nu23 = -S[0, 1] / S[0, 0] 
    nu32 = -S[1, 0] / S[1, 1]
    return E22, E33, G23, nu23, nu32

def CalcPropPlaneStress3D(C):
    """
    Calculates properties for Plane Stree.
        Args:
            C (np.ndarray): Stiffness matrix.
        Returns:
            tuple: (E11, E22, E33, G12, G13, G23, nu12, nu13, nu23)
    """
    if C.shape != (6,6):
        raise ValueError("C must be 6x6.")

    S = np.linalg.inv(C)
    E11 = 1 / S[0,0]
    E22 = 1 / S[1,1]
    E33 = 1 / S[2,2]
    nu12 = -S[0, 1] / S[0, 0]
    nu13 = -S[0, 2] / S[0, 0]
    nu23 = -S[1, 2] / S[1, 1]
    nu21 = nu12 * E22 / E11
    nu31 = nu13 * E33 / E11
    nu32 = nu23 * E33 / E22
    G23 = 1 / S[3,3]
    G13 = 1 / S[4,4]
    G12 = 1 / S[5,5]
    return {
        "E11": E11, "E22": E22, "E33": E33,
        "nu12": nu12, "nu13": nu13, "nu23": nu23,
        "nu21": nu21, "nu31": nu31, "nu32": nu32,
        "G12": G12, "G13": G13, "G23": G23
    }

def process_and_print_results(cm_matrix, sigma_matrix):
    """
    Processes CM and SIGMA matrices and prints elastic constants and stresses.
    
    Args:
        cm_matrix (np.ndarray): CM matrix extracted from FAST.
        sigma_vect(np.ndarray): Stresses matrix extracted from FAST.
    """
    print("Starting calculation of elastic properties from the FAST CM matrix...")
    print("\nMatrix CM (FAST):")
    print(cm_matrix)
    
    # --- Properties for Plane Stress - Micro Model 2D---
    if cm_matrix.shape == (3, 3):
        print("\n--- Properties for Plane Stress ---")
        E_ps, nu_ps, G1_ps, G2_ps = CalcPropPlaneStress(cm_matrix)
        print(f"E: {E_ps:.6e}")
        print(f"nu: {nu_ps:.6f}")
        print(f"G1: {G1_ps:.6e}")
        print(f"G2: {G2_ps:.6e}")

        # --- Properties for Plane Strain ---
        print("\n--- Properties for Plane Strain ---")
        E_pstrain, nu_pstrain, G1_pstrain, G2_pstrain = CalcPropPlaneStrain(cm_matrix)
        print(f"E: {E_pstrain:.6e}")
        print(f"nu: {nu_pstrain:.6f}")
        print(f"G1: {G1_pstrain:.6e}")
        print(f"G2: {G2_pstrain:.6e}")
        
        # --- Properties for Plane Strain (3x3) ---
        print("\n--- Properties for Plane Strain (3x3) ---")
        E11, E22, G12, nu12, nu21 = CalcPropPlaneStrain3x3(cm_matrix)
        print(f"E22: {E11:.6e}")
        print(f"E33: {E22:.6e}")
        print(f"G23: {G12:.6e}")
        print(f"nu23: {nu12:.6f}")
        print(f"nu32: {nu21:.6f}")
    
    # --- Properties for Plane Stress - Micro Model 3D---
    elif cm_matrix.shape == (6, 6):
        print("\n--- Properties for Plane Stress ---")
        props3D = CalcPropPlaneStress3D(cm_matrix)
        print(f"E11: {props3D['E11']:.6e}")
        print(f"E22: {props3D['E22']:.6e}")
        print(f"E33: {props3D['E33']:.6e}")
        print(f"nu12: {props3D['nu12']:.6f}")
        print(f"nu13: {props3D['nu13']:.6f}")
        print(f"nu23: {props3D['nu23']:.6f}")
        print(f"nu21: {props3D['nu21']:.6f}")
        print(f"nu31: {props3D['nu31']:.6f}")
        print(f"nu32: {props3D['nu32']:.6f}")
        print(f"G12: {props3D['G12']:.6e}")
        print(f"G13: {props3D['G13']:.6e}")
        print(f"G23: {props3D['G23']:.6e}")
    
    else:
        raise ValueError(f"CM matrix dimension not supported: {cm_matrix.shape}. Expected (3, 3) or (6, 6)..")

# Main Execution 
if __name__ == "__main__":
    # Adapt the FAST executable path and .dat file name
    fast_exe_path = r"C:\Users\Viny Pereira\Desktop\workspace\dissertacao\FASTv2.4.5\build\fast.exe"
    
    # Set the name of the .dat file you want to process
    #dat_file_name_base = 'MieheKochEx3_Ev'
    #dat_file_name_base = 'Brick20_SunVaidya_4d'
    dat_file_name_base = 'SunVaidya3D_TET10_4div'
    #dat_file_name_base = "MieheKoch_2D"
    pos_file_path = os.path.join(os.path.dirname(fast_exe_path), dat_file_name_base + ".pos")

    start_time = time.time()

    try:
        # Try to run FAST and extract CM and SIGMA from standard output
        cm, sigma = run_fast(fast_exe_path, dat_file_name_base, timeout=10000)
        process_and_print_results(cm, sigma)
    except RuntimeError as e:
        print(f"Error running FAST: {e}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "="*50)
    print(f"Total execution time: {elapsed:.2f} seconds")
    print("="*50)