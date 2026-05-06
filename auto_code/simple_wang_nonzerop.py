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
    systems_list = {"P1": 'wang_model_nonzerop'}
    constants_file = "wang_model_nonzerop." + parameter
    #parameter_lab_auto = parameter + "_scaled"
    
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
    omega = run(wang_model, NMX  = 1000000, NPR  = 100000000,
                        EPSL = 1e-05, EPSU = 1e-05, EPSS = 1e-04,
                        DS = 1e-4, DSMIN = 1e-6, DSMAX = 5e-2,
                        UZSTOP = {'P': [0.0, 2.0]},
                        JAC = 0)
    # Extract the fort.9 data
    fort9_df_fwd = parse_fort9(downsample=100)#claires code downsamples
    

    # Extract the data and export to a csv file
    omega_df = [extract_data(omega[0], data_label = ['SP'], system=system_chosen)]
    pd.concat(omega_df).to_csv(Path(data_dir,system_chosen+"_sp_data.csv"), index = False)

    
    # eigenvalues_df = fort9_df[[parameter_lab_auto, 'TY', 'stability', 'eigenvalues']]
    # eigenvalues_df.to_csv(Path(data_dir,system_chosen+"_sp_eigenvalues.csv"), index = False)
    fort9_df = fort9_df_fwd
    
    fort9_df = fort9_df.drop(columns=['eigenvalues'])
    fort9_df['system_id'] = system_chosen
    fort9_df.to_csv(Path(data_dir,system_chosen+"_sp_fort9.csv"), index = False)

    print(rf"{'-'*100}")
    print(system_chosen + " : HOPF BIFURCATION 1")
    print(rf"{'-'*100}")

    hb1_start = load(omega('HB1'), ISW=1)
    print(hb1_start)
    # Run forward
    hb1_forward = run(hb1_start, IPS=2, MXBF = 1,
                    ICP=['P', 'PERIOD'],
                    THL =  {'PERIOD': 0.10},
                    NTST = 100, NCOL = 5, NMX=400000, NPR=100000,
                    DS=5e-5, DSMAX = 1e-3,
                    EPSL = 1e-4, EPSU = 1e-4, EPSS = 1e-3,
                    SP = ['BP1'],ISW = -1,
                    JAC = 0)
    fort9_df_fwd = parse_fort9()
    # Run backwards
    hb1 = hb1_forward 
    hb1_labels = ['HB1_fwd']
    

    
    #bp1_start = load(hb1_forward('BP'), ISW=1)
    #print(bp1_start)
    
    # Run forward
    #bp1_forward = run(bp1_start, IPS=2, MXBF = 1,
    #                ICP=['V', 'PERIOD'],
    #                THL =  {'PERIOD': 0.10},
    #                NTST = 100, NCOL = 5, NMX=400000, NPR=100000,
    #                DS=0.001, DSMAX = 1e-1,
    #                EPSL = 1e-4, EPSU = 1e-4, EPSS = 1e-2,
    #                SP = ['BP1'],ISW = -1,
    #                JAC = 0)
    #fort9_df_fwd = parse_fort9()

    # Extract the data and export to a csv file
    hb1_df = [extract_data(hb1[i], data_label = hb1_labels[i], system=system_chosen) for i in range(len(hb1))]
    pd.concat(hb1_df).to_csv(Path(data_dir,system_chosen+"_hb1_data.csv"), index = False)

    # Export the fort.9 data
    #fort9_df_fwd['branch_id'] = 'HB1_fwd'
    #fort9_df_bwd['branch_id'] = 'HB1_bwd'
    #fort9_df = fort9_df_fwd #pd.concat([fort9_df_fwd, fort9_df_bwd])
    #fort9_df = fort9_df.drop(columns=['multipliers'])
    #fort9_df['system_id'] = system_chosen
    fort9_df.to_csv(Path(data_dir,system_chosen+"_hb1_fort9.csv"), index = False)


    # Run from the second hopf bifurcation point for beta1 (omega has only 1 hopf point)
    #-----------------------------------------------------------------------------------------
    if (parameter == "omega"):
        return omega + hb1
    '''
    print(rf"{'-'*100}")
    print(system_chosen + " : HOPF BIFURCATION 2")
    print(rf"{'-'*100}")
    hb2_start = load(omega('HB2'), ISW=1)
    # Run forward
    hb2_forward = run(hb2_start, IPS=2, MXBF = 1,
                    ICP=['P', 'PERIOD'],
                    THL =  {'PERIOD': 0.10},
                    NTST = 100, NCOL = 5, NMX=20000, NPR=100000,
                    DS=0.001, DSMAX = 1e-1,
                    EPSL = 1e-4, EPSU = 1e-4, EPSS = 1e-2,
                    SP = ['BP1'],
                    JAC = 0)
    fort9_df_fwd = parse_fort9()

    # Combine the solutions
    hb2 = hb2_forward 
    hb2_labels = ['HB2_fwd', 'HB2_bwd']

    # Extract the data and export to a csv file
    hb2_df = [extract_data(hb2[i], data_label = hb2_labels[i], system=system_chosen) for i in range(len(hb2))]
    pd.concat(hb2_df).to_csv(Path(data_dir, system_chosen+"_hb2_data.csv" ), index = False)

    # Export the fort.9 data (we don't need to output the multipliers data)
    #fort9_df_fwd['branch_id'] = 'HB2_fwd'
    #fort9_df_bwd['branch_id'] = 'HB2_bwd'
    #fort9_df = fort9_df_fwd #pd.concat([fort9_df_fwd, fort9_df_bwd])
    #fort9_df = fort9_df.drop(columns=['multipliers'])
    #fort9_df['system_id'] = system_chosen
    #fort9_df.to_csv(Path(data_dir,system_chosen+"_hb2_fort9.csv"), index = False)
 
    return omega + hb1 + hb2 #+bp1_forward
   
    #return omega + hb1
    
def main():
    parser = argparse.ArgumentParser(description="Run bifurcation analysis.")
    parser.add_argument('-p', '--parameter', default='P', choices=['P'], 
                        help='Parameter to use for bifurcation analysis (default: P)')
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
        #run_simple_wang(parameter, system)
        #except:
        #    print("Error occured for system: ", system)
        #    continue
    #else:
        print("Running bifurcation analysis for system: ", system_chosen)
        bd = run_simple_wang(parameter, system_chosen)
        auto.plot(bd)
        auto.wait()

    # Clean files
    auto.clean()

if __name__ == "__main__":
    # run_1d_bifurcation("beta1", "P1")
    main()