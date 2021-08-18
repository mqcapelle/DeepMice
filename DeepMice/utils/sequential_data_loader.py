import numpy as np
import xarray as xr

from torch.utils.data import DataLoader, Dataset
import torch

from DeepMice.utils.helpers import set_seed, seed_worker


def get_train_val_test_loader(path, len_seq=15, batch_size=10, SEED=None):
    # Load data
    data_interface = DataInterface(path=path)

    # prepare datasets
    dataset_train = SequentialDataset(data_interface, part='train', len_sequence=len_seq)
    dataset_val = SequentialDataset(data_interface, part='val', len_sequence=len_seq, )
    dataset_test = SequentialDataset(data_interface, part='test', len_sequence=len_seq, )

    # set seeds for everything
    g_seed = torch.Generator()
    g_seed.manual_seed(SEED)
    set_seed(seed=SEED)

    # initialize loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              worker_init_fn=seed_worker, generator=g_seed)

    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                            worker_init_fn=seed_worker, generator=g_seed)

    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                             worker_init_fn=seed_worker, generator=g_seed)

    return train_loader, val_loader, test_loader


class DataInterface:
    """ Interface for train, test and val data from xarray file
  
    Initialize with:
       data_interface = DataInterface(path_to_xarray_file)
      
    Get a subset of the data with:
        subset = data_interface.get_subset(part='train')   # 'train', 'test', or 'val'
       
    """
    def __init__(self, path=None):
        """ If path=None, then dummy data is loaded """
        # Assign parameters
        if path is not None:
            self.path = path
            self.data = xr.open_dataset(path)
            self.trial_mat, self.y, self.time = self.get_trial_matrix_3d(self.data)
        else:
            # for testing
            self.trial_mat = np.zeros( (50, 1, 1) )
            self.trial_mat[:,0,0] = np.arange(50)
            self.y = np.arange(50)*10
            self.time = np.arange(50)/10

        self.nr_trials = self.trial_mat.shape[0]
        self.masks = self.get_split_masks(self.nr_trials)

    def get_subset(self, part='train'):
        """ Main interface to return the train, test or val set of the dataset
        
        Choices for part: 'train', 'test' or 'val'
        """
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


class SequentialDataset:
    """ Get sequences of a few trials of neural activity, images and initial image.
    
    Create a sequentail dataset:
        data_interface = DataInterface(path_to_xarray_file)
        dataset = SequentialDataset(data_interface,
                                    part = 'train',
                                    len_sequence = 10 )
    
    Use dataset then in a DataLoader for access to batches of datasets
        train_loader = DataLoader(dataset, batch_size=5, shuffle=True,
                                  worker_init_fn = 2342)
        X_batch, y_batch, init_batch = next( iter(train_loader) )
        
    
    """
    def __init__(self, data_interface,
                 part='train',
                 len_sequence=10):
        
        subset = data_interface.get_subset(part)
        
        self.X = subset[0]
        self.y = subset[1]
        self.t = subset[2]
        self.len_seq = len_sequence
        
        # valid_index are the index values which return continous time chunks
        # of length len_sequence without out-of-bounds errors
        self.valid_index = self.calculate_valid_index(self.t, self.len_seq)
        
    
    def __getitem__(self, index):
        val_ind = self.valid_index[index]
        X_part = self.X[val_ind:val_ind+self.len_seq, :, :]
        y_part = self.y[val_ind:val_ind+self.len_seq]
        init = self.y[val_ind-1]   # starting value for sequence
        
        # shape of return values
        # X_part: (len_sequence, neurons, time)     Input data of neural activities
        # y_part: (len_sequence)                    Image number of the dataset
        # init:   (1)                               Previous image before start of sequence
        
        return torch.Tensor(X_part), torch.Tensor(y_part), torch.Tensor([init])
    
    
    def __len__(self):
        """Returns the number """
        return len( self.valid_index )
    
    
    def calculate_valid_index(self, t, len_seq):
        """ Calculate indicies that start continous time series and has an initial value """
        valid_index = list()
        
        index = np.arange( len(t) ).astype(int)  # all indicies of t
        index = index[1:-len_seq]   # remove last len_seq elements
        
        for i in index:  
            # select time chunk starting one before i and until end of sequence
            t_part = t[i-1:i+len_seq]
            
            # calculate maximum time difference in the chunk
            max_diff = np.max( np.diff(t_part) )
            
            # append the index if the chunk is continour in time
            if max_diff < 1:
                valid_index.append(i)
                    
        return np.array( valid_index )
       
        
if __name__ == '__main__':
    ex_path = '020_excSession_v1_ophys_858863712.nc'    # example file
    
    data_interface = DataInterface(path=ex_path)
    dataset = SequentialDataset(data_interface,
                                part='train',    # or 'test', 'val'
                                len_sequence=15
                                ) 
    
    train_loader = DataLoader(dataset,
                          batch_size=5,
                          shuffle=True,
                          worker_init_fn=3453)
    
    X_b, y_b, init_b = next( iter(train_loader) )
    
    print(X_b.shape,    # Neural activity (batch_size, len_sequence, nr_neurons, time)
          y_b.shape,    # Shown images    (batch_size, len_sequence)
          init_b.shape) # Initial image   (batch_size, 1)
