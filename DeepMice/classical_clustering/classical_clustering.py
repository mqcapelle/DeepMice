
# ######################################
# Imports
# ######################################
# Standard library imports
from pathlib import Path
# Third party imports
from DeepMice.classical_clustering.ephys_extractor import EphysSweepFeatureExtractor
import xarray as xr

# Local library imports
from DeepMice.utils.data_loader import easy_train_test_loader()

# ######################################
# Paths
# ######################################
path_to_data_folder = Path("DeepMice/data")

for path_to_data_file in path_to_data_folder.glob("*.nc"):
    print(f"{path_to_data_file} ...")
    # ######################################
    # Load data
    # ######################################
    data = xr.open_dataset(path_to_data_folder.joinpath(path_to_data_file))


# ######################################
# Extract features
# ######################################








