# Libraries
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os
# Must import parse_auto before we import auto
from parse_auto import parse_fort9, extract_data
import auto
from auto import run, load

def run_simple_wang(parameter, system_chosen):
    # Dictionary to map the system and parameter to the correct model and parameter names
    systems_list = {"P1": 'wang_model'}
    constants_file = "wang_model." + parameter
    parameter_lab_auto = parameter + "_scaled"
    
    # Data directory (extra .. as we change directories above)
    data_dir = Path("../output/bifurcation",parameter)
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the model - files: immune_model.f90 and c.immune_model
    wang_model = load(systems_list[system_chosen], c=constants_file)
    
    print(rf"{'-'*100}")
    print(system_chosen + " : STABLE POINTS")
    print(rf"{'-'*100}")
    
    # Run the 1D bifurcation (omega) forwards
    omega_forward = run(wang_model, NMX  = 1000000, NPR  = 100000000,
                        EPSL = 1e-05, EPSU = 1e-05, EPSS = 1e-04,
                        DS = 1e-6, DSMIN = 1e-8, DSMAX = 5e-4,
                        UZSTOP = {'V': [-100, 0]},
                        JAC = 0)
    # Extract the fort.9 data
    fort9_df_fwd = parse_fort9(downsample=100)#claires code downsamples
    
    
    
    
def main():
    parser = argparse.ArgumentParser(description="Run bifurcation analysis.")
    parser.add_argument('-p', '--parameter', default='V', choices=['V'], 
                        help='Parameter to use for bifurcation analysis (default: V)')
    parser.add_argument('-s', '--system', default='P1', choices=['P1'],
                        help='System (defined by .f90 files) to use for bifurcation analysis (degault: P1)')
    parser.add_argument('--all', action='store_true', 
                        help='Use all systems for bifurcation analysis for parameter in -p')

    # Get the specified parameters
    args = parser.parse_args()
    parameter = args.parameter
    system_chosen = args.system

    # Change to the correct working directory containing the auto files
    auto_files_dir = "simple_wang_auto_files/"
    os.chdir(auto_files_dir)

    ## Run the bifurcation analysis

    for system in ['P1']:
        print("Running bifurcation analysis for system: ", system)
        # Continue to next system even if there is an error
        #ry:
        run_simple_wang(parameter, system)
        #except:
        #    print("Error occured for system: ", system)
        #    continue
    #else:
    #    print("Running bifurcation analysis for system: ", system_chosen)
    #    bd = run_simple_wang(parameter, system_chosen)
    #    auto.plot(bd)
    #    auto.wait()

    # Clean files
    auto.clean()

if __name__ == "__main__":
    # run_1d_bifurcation("beta1", "P1")
    main()