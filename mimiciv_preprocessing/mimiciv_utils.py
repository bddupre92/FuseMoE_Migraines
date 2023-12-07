import json
import os

# Get parent of file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(parent_dir, 'file_paths.json'), 'r') as f:
    file_paths = json.load(f)

    # If file_paths['mimic_iv_data'] folder does not exist, create it
    if not os.path.exists(file_paths['mimiciv_data']):
        os.makedirs(file_paths['mimiciv_data'])
    
    file_paths["preprocessing"] = os.path.join(file_paths["mimiciv_data"], "preprocessing")

    # If file_paths['preprocessing'] folder does not exist, create it
    if not os.path.exists(file_paths['preprocessing']):
        os.makedirs(file_paths['preprocessing'])

def get_path(folder):
    """
    Returns the path to the folder specified in the file_paths.json file
    """
    return file_paths[folder]