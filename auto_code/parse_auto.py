import re
import pandas as pd
from pathlib import Path

# AUTO-07p python setup
#------------------------
import sys, os
auto_dir = "auto-07p" # relative path to auto program directory
auto_dir = Path(auto_dir).resolve() # Safest to pass as full path
os.environ['AUTO_DIR'] = str(auto_dir)
sys.path.append(Path(auto_dir,"bin"))
sys.path.append(Path(auto_dir,"cmds"))

# Functions for processing auto output
#--------------------------------------

# Function to read and write the data from a bifDiagBranch object
def extract_data(branch, data_label, system = None):
        header = branch.keys()
        point_types = [pt['TY'] if pt['TY'] != 'No Label' else None
                    for pt in branch]
        df = pd.DataFrame(branch.toArray(), columns=header)
        df['point_type'] = point_types
        if (system is not None):
            df['system_id'] = system
        df['branch_id'] = data_label
        return(df)

# Function to parse the data in the fort9 files
def parse_fort9(include_line_numbers = False, downsample = None):

    with open('fort.9', 'r') as file:
        lines = list(file.readlines())

    # Things are a bit easier to follow if we do it in reversed order
    lines_rev = list(reversed(lines))
    nlines = len(lines_rev)

    # Find the lines that define the points: 
    # we assume these precede (in reversed order) a header line starting with 'BR    PT'
    pts_idx = [i-1 for i, line in enumerate(lines_rev) if bool(re.search(r'BR\s+PT',line))]
    npoints = len(pts_idx)
    print(f'Found {npoints} points in {nlines} lines')

    # Loop over points and store their branch and point id, and their stability
    points_info = [None]*npoints
    for i, curr_point_idx in enumerate(pts_idx):
        # Extract the point information from the line above and current line
        values_list = lines_rev[curr_point_idx].split()
        names_list = lines_rev[curr_point_idx+1].replace("MAX ","MAX.").split()
        # Often missing the value of TY (3rd column) so add it in
        if len(values_list) < len(names_list):
            values_list.insert(2, None)
        variables_dict = dict(zip(names_list, values_list))
        if include_line_numbers:
            variables_dict['pt_info_line_num'] = nlines-curr_point_idx

        # For some reason sometimes the points and branches have a '-' in them
        variables_dict['BR'] = variables_dict['BR'].replace('-', '')
        variables_dict['PT'] = variables_dict['PT'].replace('-', '')

        # Loop over the point information lines
        next_pt_idx = pts_idx[i+1] if (i+1) < npoints else nlines
        eigenvalues = []
        multipliers = []
        for j in range(curr_point_idx, next_pt_idx):
            # Search for first 'Stable:' after each point
            if 'Stable:' in lines_rev[j]:
                line_search = re.search(r'(\d+)\s+(\d+)\s+(Multipliers|Eigenvalues\s+):\s+Stable:\s+(\d+)', lines_rev[j])
                # Check the branch and point ids (for some reason one is negative?) match
                assert line_search.group(1) == variables_dict['BR'],  f'BR error: {line_search.group(1)} != {variables_dict["BR"]} for point {variables_dict["PT"]} at line {nlines-curr_point_idx}'
                if variables_dict['PT'] != '1': # For some reason the point can be 1 and the iteration 1000
                    assert line_search.group(2) == variables_dict['PT'], f'PT error: {line_search.group(2)} != {variables_dict["PT"]} for point {variables_dict["PT"]} at line {nlines-curr_point_idx}'
                variables_dict['stability'] = line_search.group(4)
                if eigenvalues:
                    variables_dict['eigenvalues'] = eigenvalues
                if multipliers:
                    variables_dict['multipliers'] = multipliers
                if include_line_numbers:
                    variables_dict['stability_line_num'] = nlines-j
                
                break
            # Also store the eigenvalues/multipliers as we go, in correct order (always precede the stable line)
            if 'Eigenvalue' in lines_rev[j]:
                line_search = re.search(r'Eigenvalue\s+\d+:\s+(.*)', lines_rev[j])
                eigenvalues = [line_search.group(1), *eigenvalues]
            if 'Multiplier' in lines_rev[j]:
                line_search = re.search(r'Multiplier\s+\d+\s+(.*)', lines_rev[j])
                multipliers = [line_search.group(1), *multipliers]

        # Store the variables in the output list
        points_info[i] = variables_dict

    output_df = pd.DataFrame(reversed(points_info))

    # Downsample but keep rows that have a non-empty value in the column 'TY'
    if downsample:
        print("Downsampling...")
        print(output_df.shape)
        output_df = output_df[(output_df.index % downsample == 0) | (output_df['TY'].notna())]
        print(output_df.shape)

    return(output_df)

# parse_fort9()