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
                        output='image_index',
                        include_time=False):
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
  time = np.zeros( (nr_trials, nr_frames) )
  y = np.zeros( nr_trials )
    
  for i in range(nr_trials):
    # extract the neural activity 
    start_idx = int( data.start_frame[i] )   # frame of trial start
    trial_matrix_3d[i,:,:] = data.activity.data[:,start_idx:start_idx+nr_frames]

    # extract time
    time[i,:] = data.activity.time[start_idx:start_idx+nr_frames]
    
    # select the predictor that should be used
    if output == 'image_index':
      y[i] = data.image_index[i]
    elif output == 'is_change':
      y[i] = data.is_change[i]
    elif output == 'rewarded':
      y[i] = data.rewarded[i]
    else:
      raise Exception('Argument for output="{}" not supported.'.format(output))
  
  if not include_time:
    return trial_matrix_3d, y
  else:
    return trial_matrix_3d, y, time


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
                           test_ratio=0.2, split_type='block_middle',
                           with_time=True, return_all=False):
  """ Get train and test data loader from data xarray
  
  TODO: documentation, for now check called functions
  """
  # find out how many frames still belong to one trial (0.75 seconds)
  nr_frames_after = int( data.attrs['frame_rate_Hz'] * 0.7 )

  # get cut out pieces of activity in mat_3d for each trial
  mat_3d, y, t = get_trial_matrix_3d(
                    data=data,
                    nr_frames_after=nr_frames_after,
                    output=output, include_time=True)

  # mat_3c (trials, neurons, time)
  if with_time:
    X = mat_3d
  else:
    # average out time for now for compatibility across sampling rates
    X = np.mean(mat_3d, axis=2)

  # get masks with True/False for train/test trials
  nr_trials = X.shape[0]
  train_mask, test_mask = get_train_test_mask(nr_trials=nr_trials,
                                              split_type = split_type,
                                              ratio = test_ratio)

  train_loader, test_loader = get_train_test_loader(
        X=X, y=y, train_mask=train_mask, test_mask=test_mask, batch_size=batch_size)
  
  if not return_all:
    return train_loader, test_loader
  else:
    return train_loader, test_loader, X, y, t, train_mask, test_mask

def easy_train_test_loader_fixed_size_per_session(
    path_to_data_file,
    batch_size=128, output='image_index',
    test_ratio=0.2, split_type='block_middle',
    with_time=True, return_all=False,
    min_number_of_neurons=400, max_number_of_neurons=1000,
    max_number_of_timestamps=False,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng(seed=2021)

    # Load data
    data = xr.open_dataset(path_to_data_file)

    # Only use behavior sessions with correct amount of neurons
    number_of_neurons = len(data.neuron_id)
    if number_of_neurons < min_number_of_neurons or number_of_neurons > max_number_of_neurons:
        warnings.warn(f"Number of neurons ({number_of_neurons}) not sufficient for {path_to_data_file}")

    # Load test and train data
    _, _, x, y, t, train_mask, test_mask = easy_train_test_loader(
        data=data,
        batch_size=batch_size,
        output=output,
        test_ratio=test_ratio,
        split_type=split_type,
        with_time=with_time,
        return_all=return_all,
    )

    for usage in ["test", "train"]:
        exec(f"x_{usage} = x[{usage}_mask]")
        exec(f"y_{usage} = y[{usage}_mask]")
        exec(f"t_{usage} = t[{usage}_mask]")

        # Shuffle over neuron positions
        # TODO: maybe implement this

        # Stack to new shape
        exec(f"(x_size_trials, x_size_neurons, x_size_time) = np.shape(x_{usage})")
        # # Check if max_number_of_timestamps is constant
        if max_number_of_timestamps and x_size_time != max_number_of_timestamps:
            warnings.warn(f"Number of time stamps ({x_size_time}) is not equal to "
                          f"maximum_number_of_timestamps ({max_number_of_timestamps})")

        # Stack to max_number_of_neurons and max_number_of_time_stamps
        x_new_shape = np.zeros(shape=(x_size_trials, max_number_of_neurons, max_number_of_timestamps))
        exec(f"x_new_shape[:, :x_size_neurons, :x_size_time] = x_{usage}")
        exec(f"x_{usage} = x_new_shape")

    return x_test, y_test, t_test, x_train, y_train, _t_train
    

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



