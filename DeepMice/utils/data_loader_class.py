# Standard library imports
from pathlib import Path
import os
import requests
import warnings

# Third party library imports
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class DeepMiceDataLoader:
    def __init__(
            self,
            path_to_data_folder,
            batch_size=128, output='image_index',
            test_ratio=0.2, split_type='block_middle',
            with_time=True, return_all=False,
            min_number_of_neurons=400, max_number_of_neurons=1000,
            max_number_of_timestamps=None, max_number_of_sessions=None,
            shuffle_trials=False, shuffle_neurons=False,
            rng=None
    ):

        self.path_to_data_folder = path_to_data_folder
        self.path_to_data_files = list(self.path_to_data_folder.glob("*.nc"))

        self.batch_size = batch_size
        self.test_ratio = test_ratio

        self.output = output
        self.split_type = split_type
        self.with_time = with_time
        self.return_all = return_all

        self.min_number_of_neurons = min_number_of_neurons
        self.max_number_of_neurons = max_number_of_neurons
        self.max_number_of_timestamps = max_number_of_timestamps
        self.max_number_of_sessions = max_number_of_sessions


        self.shuffle_trials = shuffle_trials
        self.shuffle_neurons = shuffle_neurons

        if rng is None:
            self.rng = np.random.default_rng(seed=2021)

    def load_single_session(self, path=None):
        """ Load one session from file

        Input:
          path: str, optional (path to the downloaded file)
        Output:
          data: xarray with data for one session
        """

        if path is None:
            path = self.path_to_data_files[0]

        return xr.open_dataset(path)

    def easy_train_test_loader(self, data, trial_length=0.7):
        """
        TODO: Documentation
        """
        # find out how many frames still belong to one trial (trial_length seconds)
        nr_frames_after = int(data.attrs['frame_rate_Hz'] * trial_length)

        # get cut out pieces of activity in mat_3d for each trial
        X, y, t = self.get_trial_matrix_3d(data=data, nr_frames_after=nr_frames_after)

        # Average out time for compatibility across sampling rates
        if self.with_time:
            X = np.mean(X, axis=2)

        # Split data in train and test trials
        nr_trials = X.shape[0]
        train_mask, test_mask = self.get_train_test_mask(nr_trials=nr_trials)

        train_loader, test_loader = self.get_train_test_loader(X, y, train_mask, test_mask)



    def get_trial_matrix_3d(self, data, nr_frames_after):
        """Calculate 3D trial matrix (trials,neurons,time) from loaded data
        Input:
          data: xarray for one session
          nr_frames_after: int   (frames to extract after trial onset)
          output: str   (description of the type of variable)
                Supported choices: image_index, is_change, rewarded

        Output:
          trial_matrix_3d: (trials, neurons, time)
          y: 1d array (trials)
        """
        # use descriptive variables for the dimensions
        nr_trials = len(data.trial)
        nr_frames = nr_frames_after
        nr_neurons = data.activity.shape[0]

        trial_matrix_3d = np.zeros((nr_trials, nr_neurons, nr_frames))
        time = np.zeros((nr_trials, nr_frames))
        y = np.zeros(nr_trials)

        for i in range(nr_trials):
            # extract the neural activity
            start_idx = int(data.start_frame[i])  # frame of trial start
            trial_matrix_3d[i, :, :] = data.activity.data[:, start_idx:start_idx + nr_frames]

            # extract time
            time[i, :] = data.activity.time[start_idx:start_idx + nr_frames]

            # select the predictor that should be used
            if self.output == 'image_index':
                y[i] = data.image_index[i]
            elif self.output == 'is_change':
                y[i] = data.is_change[i]
            elif self.output == 'rewarded':
                y[i] = data.rewarded[i]
            else:
                raise Exception('Argument for output="{}" not supported.'.format(self.output))

        return trial_matrix_3d, y, time

    def get_train_test_mask(self, nr_trials):
        """ Return two masks with True/False to select train and test trials
            required to split data into train and test set
        """
        # Test mask
        test_mask = np.zeros(nr_trials) > 0  # all False
        if self.split_type == 'block_middle':
            test_mask[int((0.5 - self.test_ratio / 2) * nr_trials):int((0.5 + self.test_ratio / 2) * nr_trials)] = True
        elif self.split_type == 'random':
            test_mask[0:int(self.test_ratio * nr_trials)] = True
            test_mask = self.rng.permutation(test_mask)
        else:
            raise Exception('Split type "{}" not implemented'.format(self.split_type))

        # Train mask
        train_mask = (test_mask == False)  # invert mask
        return train_mask, test_mask

    def train_test_loader(self, max_number_of_sessions):
        x_data = None

        session_counter = 0
        for path_to_data_file in tqdm(self.path_to_data_files, desc="Load data"):
            # Load data
            data = xr.open_dataset(path_to_data_file)

            # Only use behavior sessions with correct amount of neurons
            number_of_neurons = len(data.neuron_id)
            if number_of_neurons < self.min_number_of_neurons or number_of_neurons > max_number_of_neurons:
                continue

            # Stop if max_number_of_sessions is reached
            session_counter += 1
            if max_number_of_sessions and session_counter > max_number_of_sessions:
                break