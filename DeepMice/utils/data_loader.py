# Standard library imports
from pathlib import Path
import os
import requests

# Third party library imports
import numpy as np
import xarray as xr


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
