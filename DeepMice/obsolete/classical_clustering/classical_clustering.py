
# ##############################################################
# Imports
# ##############################################################
# Standard library imports
from pathlib import Path
# Third party imports
import matplotlib.pyplot as plt

# Local library imports
from DeepMice.obsolete.data_loader import easy_train_test_loader, load_one_session

# ##############################################################
# Paths
# ##############################################################
path_to_data_folder = Path("/DeepMice/data")
path_to_data_files = list(path_to_data_folder.glob("*.nc"))
path_to_data_file = path_to_data_files[1]

# ##############################################################
# Load data
# ##############################################################
print(f"{path_to_data_file} ...")
data = load_one_session(path_to_data_file)
_, _, x_batch, y_batch, t_batch, train_mask, test_mask = easy_train_test_loader(
    data=data,
    batch_size=128,
    output='is_change',
    test_ratio=0.2,
    split_type='block_middle',
    return_all=True,
)
x_batch, y_batch, t_batch = x_batch[train_mask], y_batch[train_mask], t_batch[train_mask]

# ##############################################################
# Extract features
# ##############################################################
for (x, t, y) in zip(x_batch[0:20], t_batch, y_batch):  # loop over trials
    plt.figure()
    plt.title(y)
    for v in x:  # loop over neurons
        # f, f_names, f_units = calc_features(
        #     v=v, t=t
        # )
        # f_dict = dict(zip(f_names, f))
        # print(f_dict)

        plt.plot(t, v)
# plt.show()

# plt.plot(data.activity.time, data.activity[0])

