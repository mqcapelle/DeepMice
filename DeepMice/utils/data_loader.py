# Standard library imports
from pathlib import Path
import os
import requests

# Third party library imports
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, TensorDataset

def download_data(fdir=None):
    # TODO: rewrite this function to download data from OUR drive
    raise Exception("Code not implemented yet!")

    url = 'https://ndownloader.figshare.com/files/28470255'
    r = requests.get(url, allow_redirects=True)

    if fdir is None:
        fdir = os.getcwd()
    assert isinstance(fdir, str)

    filename = "allen_visual_behavior_2p_change_detection_" \
               "familiar_novel_image_sets.parquet"
    # data = pd.read_parquet(filename)

    print('Downloading...')
    open(op.join(fdir, filename), 'wb').write(r.content)
    print('Done!\nSaved at {0}'.format(op.join(fdir, filename)))


def load_one_session(path='046_excSession_v1_ophys_971632311.nc'):
  """ Load one session from file

  Input:
    path: str (path to the downloaded file)
  Output:
    data: xarray with data for one session
  """
  data = xr.open_dataset(path)
  return data


def get_trial_matrix_3d(data, nr_frames_after=10,
                        output='image_index'):
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
  nr_trials = len( data.trial )
  nr_frames = nr_frames_after
  nr_neurons = data.activity.shape[0]

  trial_matrix_3d = np.zeros( (nr_trials, nr_neurons, nr_frames))
  y = np.zeros( nr_trials )

  for i in range(nr_trials):
    # extract the neural activity 
    start_idx = int( data.start_frame[i] )   # frame of trial start
    trial_matrix_3d[i,:,:] = data.activity.data[:,start_idx:start_idx+nr_frames]

    # select the predictor that should be used
    if output == 'image_index':
      y[i] = data.image_index[i]
    elif output == 'is_change':
      y[i] = data.is_change[i]
    elif output == 'rewarded':
      y[i] = data.rewarded[i]
    else:
      raise Exception('Argument for output="{}" not supported.'.format(output))
  
  return trial_matrix_3d, y



def get_train_test_mask(nr_trials, split_type='block_middle',
                        ratio=0.2, seed=483982):
  """ Return two masks with True/False to select train and test trials

  """
  # split data into train and test set
  
  test_mask = np.zeros( nr_trials ) > 0   # all False

  if split_type == 'block_middle':
    test_mask[int((0.5-ratio/2)*nr_trials):int((0.5+ratio/2)*nr_trials)] = True
  elif split_type == 'random':
    test_mask[0:int(ratio*nr_trials)] = True
    np.random.seed(seed)
    test_mask = np.random.permutation(test_mask)
  else:
    raise Exception('Split type "{}" not implemented'.format(split_type))

  train_mask = (test_mask == False)    # invert mask

  return train_mask, test_mask


def get_train_test_loader(X, y, train_mask, test_mask,
                           batch_size=128, seed=7987542):
  """Get train and test loaders 
  
  Input:
    X: (nr_trials, ...)
    y: (nr_trials)
    train_mask: (nr_trials)   boolean array with True for train trials
    test_mask: (nr_trials)    like train_mask, but for test set
    
  Output:
    train_loader
    test_loader
  """

  # initialize loaders
  batch_size = 128
  g_seed = torch.Generator()
  g_seed.manual_seed(seed)

  train_data = TensorDataset(torch.from_numpy(X[train_mask]), torch.from_numpy(y[train_mask]))
  train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              worker_init_fn=seed,
                              generator=g_seed)
  
  test_data = TensorDataset(torch.from_numpy(X[test_mask]), torch.from_numpy(y[test_mask]))
  test_loader = DataLoader(test_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              worker_init_fn=seed,
                              generator=g_seed)

  return train_loader, test_loader 

def easy_train_test_loader(data, batch_size=128, output='image_index',
                           test_ratio=0.2, split_type='block_middle'):
  """ Get train and test data loader from data xarray
  
  TODO: documentation, for now check called functions
  """
  # find out how many frames still belong to one trial (0.75 seconds)
  nr_frames_after = int( data.attrs['frame_rate_Hz'] * 0.7 )

  # get cut out pieces of activity in mat_3d for each trial
  mat_3d, y = get_trial_matrix_3d(
                    data=data,
                    nr_frames_after=nr_frames_after,
                    output=output )
  # mat_3c (trials, neurons, time)
  # average out time for now for compatibility across sampling rates
  X = np.mean(mat_3d, axis=2)

  # get masks with True/False for train/test trials
  nr_trials = X.shape[0]
  train_mask, test_mask = get_train_test_mask(nr_trials=nr_trials,
                                              split_type = split_type,
                                              ratio = test_ratio)

  train_loader, test_loader = get_train_test_loader(
        X=X, y=y, train_mask=train_mask, test_mask=test_mask)
  
  return train_loader, test_loader


if __name__ == '__main__':

    # example use of the data loader (assumes file to be in working directory)
    path = '046_excSession_v1_ophys_971632311.nc'
    if not os.path.isfile(path):
      raise Exception('Example file "{}" is not in working directory.'.format(path))
    data = load_one_session(path)

    train_loader, test_loader = easy_train_test_loader( data=data,
                                                   batch_size=128,
                                                   output='image_index',
                                                   test_ratio = 0.2,
                                                   split_type = 'block_middle',
                                                   )

    X_batch, y_batch = next( iter(train_loader))
    print('Data loaded successfully :)')
    print('X shape:', X_batch.shape)
    print('y shape:', y_batch.shape)



