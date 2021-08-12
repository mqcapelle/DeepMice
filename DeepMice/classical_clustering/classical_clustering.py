
# ##############################################################
# Imports
# ##############################################################
# Standard library imports
from pathlib import Path
# Third party imports
import matplotlib.pyplot as plt
import xarray as xr

# Local library imports
from DeepMice.classical_clustering.fa_calcfeatures import calc_features
from DeepMice.utils.data_loader import easy_train_test_loader, load_one_session

# ##############################################################
# Paths
# ##############################################################
path_to_data_folder = Path("/Users/mc/PycharmProjects/DeepMice/DeepMice/data")
path_to_data_files = list(path_to_data_folder.glob("*.nc"))
path_to_data_file = path_to_data_files[0]

# ##############################################################
# Load data
# ##############################################################
print(f"{path_to_data_file} ...")
data = load_one_session(path_to_data_file)
train_loader, _ = easy_train_test_loader(
    data=data,
    batch_size=128,
    output='image_index',
    test_ratio=0.2,
    split_type='block_middle',
)
x_batches, _ = next(iter(train_loader))

# ##############################################################
# Extract features
# ##############################################################
# for x_batch in x_batches:
#     v = x_batch
#
#     f, f_names, f_compartments, f_units = calc_features(
#         v=x_batch,
#         t=
#     )