#
# # ##############################################################
# # Imports
# # ##############################################################
# # Standard library imports
# from pathlib import Path
# import warnings
# # Third party imports
# import numpy as np
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import xarray as xr
# # Local library imports
# from DeepMice.utils.data_loader import easy_train_test_loader
# from DeepMice.obsolete.classical_clustering.modern_clustering.model import set_device, Net
#
# # ##############################################################
# # Paths
# # ##############################################################
# path_to_data_folder = Path("/DeepMice/data")
# path_to_data_files = list(path_to_data_folder.glob("*.nc"))
#
# # ##############################################################
# # Settings
# # ##############################################################
# rng = np.random.default_rng(seed=2021)
#
# minimal_number_of_neurons = 400
# maximum_number_of_neurons = 1000
# maximum_number_of_time_stamps = 7
#
# x_data = np.empty(shape=(1, maximum_number_of_neurons, maximum_number_of_time_stamps))
#
# # ##############################################################
# # Load and prepare data from all behaviour sessions with correct amount of neurons
# # ##############################################################
# counter = 0
# for path_to_data_file in path_to_data_files:
#     # Load data
#     data = xr.open_dataset(path_to_data_file)
#
#     # Only use behavior sessions with correct amount of neurons
#     number_of_neurons = len(data.neuron_id)
#     if number_of_neurons < minimal_number_of_neurons or number_of_neurons > maximum_number_of_neurons:
#         continue
#
#     # Get test and train data
#     _, _, x, y, t, train_mask, test_mask = easy_train_test_loader(
#         data=data,
#         batch_size=128,
#         output='is_change',
#         test_ratio=0.2,
#         split_type='block_middle',
#         return_all=True,
#     )
#     x, y, t = x[train_mask], y[train_mask], t[train_mask]
#
#     # shuffle over neuron positions
#     rng.shuffle(x, axis=1)
#
#     # stack to maximum_number_of_neurons
#     (x_size_trials, x_size_neurons, x_size_time) = np.shape(x)
#     if x_size_time != maximum_number_of_time_stamps:
#         warnings.warn(f"Number of time stamps ({x_size_time}) is not equal to maximum_number_of_time_stamps ({maximum_number_of_time_stamps})")
#
#     x_new_shape = np.zeros(shape=(x_size_trials, maximum_number_of_neurons, maximum_number_of_time_stamps))
#     x_new_shape[:, :x_size_neurons, :x_size_time] = x
#     x_data = np.append(x_data, x_new_shape, axis=0)
#
#     print(f"{counter}; {np.shape(x)}")
#
#     counter += 1
#     if counter > 5:
#         break
#
# # Generate batches
# train_data = None
#
# # x_batch = DataLoader(train_data,
# #                               batch_size=batch_size,
# #                               shuffle=True,
# #                               num_workers=0,
# #                               worker_init_fn=seed,
# #                               generator=g_seed)
#
#
#
# # ##############################################################
# # Feed data into neural network
# # ##############################################################
# DEVICE = set_device()
# model = Net(maximum_number_of_neurons, maximum_number_of_time_stamps).to(DEVICE)
# optimizer = optim.Adadelta(model.parameters(), lr=0.1)
#
# for epoch in tqdm(range(100)):
#
#
#
#
#
