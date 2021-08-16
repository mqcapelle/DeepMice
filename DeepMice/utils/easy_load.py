import numpy as np
import xarray as xr

import torch
from torch.utils.data import DataLoader, Dataset


class DataHandler:
    def __init__(self, path):
        self.path = path
        # self.data = xr.open_dataset(path)
        # self.trial_mat, self.y, self.time = self.get_trial_matrix_3d(self.data)
        # for testing
        self.trial_mat = np.zeros( 50, 1, 1)
        self.trial_mat[:,0,0] = np.arange(50)
        self.y = np.arange(50)*10
        self.time = np.arange(50)/10

        self.nr_trials = self.trial_matrix_3d.shape[0]

        self.masks = self.get_split_masks(self.nr_trials)

    def get_subset(self, part='train'):
        m = self.masks[part]
        return self.trial_mat[m], self.y[m], self.time[m]

    # local helper functions
    def get_trial_matrix_3d(self, data, seconds_after=0.7):
        """Calculate 3D trial matrix (trials,neurons,time) from loaded data
        Input:
          data: xarray for one session
          seconds_after: float   (number of seconds to extract after trial onset)

        Output:
          trial_matrix_3d: (trials, neurons, time)
          y: 1d array (trials)
        """
        # use descriptive variables for the dimensions
        nr_trials = len(data.trial)
        nr_frames = int(data.attrs['frame_rate_Hz'] * seconds_after)
        nr_neurons = data.activity.shape[0]

        trial_matrix_3d = np.zeros((nr_trials, nr_neurons, nr_frames))
        time = np.zeros((nr_trials, nr_frames))
        y = np.zeros(nr_trials)

        for i in range(nr_trials):
            # extract the neural activity
            start_idx = int(data.start_frame[i])  # frame of trial start
            trial_matrix_3d[i, :, :] = data.activity.data[:, start_idx:start_idx + nr_frames]

            time[i, :] = data.activity.time[start_idx:start_idx + nr_frames]

            y[i] = data.image_index[i]

        return trial_matrix_3d, y, time

    def get_split_masks(self, nr_trials):
        masks = dict()
        # test
        tmp = np.zeros( nr_trials ) > 0   # all False
        tmp[int(0.35*nr_trials):int(0.5*nr_trials)] = True
        masks['test'] = tmp
        
        # validate
        tmp = np.zeros(nr_trials) > 0  # all False
        tmp[int(0.5 * nr_trials):int(0.65 * nr_trials)] = True
        masks['val'] = tmp
        
        # train
        tmp = np.zeros(nr_trials) > 0  # all False
        tmp = tmp + masks['test'] + masks['val']
        tmp = (tmp == 0)  # flip the mask
        masks['train'] = tmp

        return masks
        
        


class EasyDataset(Dataset):
    def __init__(self, path):



    def __getitem__(self, index):
        return self.data[index, :, :], self.label[index, :, :]

    def __len__(self):
        return self.data.shape[2]






