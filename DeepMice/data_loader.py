import numpy as np


def load_example_data(path='one_example_session.npy', print_keys=True):
  """ Example function to load the exported data
  Input:
    path: str (path to the downloaded file)
  Output:
    data: dict ()
  """

  data = np.load('one_example_session.npy', allow_pickle=True).item()

  if print_keys:
    print('Keys in the data dictionary:\n', data.keys() )

  return data


def get_trial_matrix_3d(activity, neuron_time, stimulus_details,
                        nr_frames_after=10):
  """Calculate 3D trial matrix (trials,neurons,time) from loaded data
  Input:
    activity: 2d matrix (neurons, time)
    neuron_time: 1d matrix (time)
    stimulus_details: pandas dataframe (as loaded from one_example_session.npy)
  Output:
    trial_matrix_3d: (trials, neurons, time)
    image_idx: (trials) Image number that is shown (8 for omitted)
  """

  nr_trials = len( stimulus_details ) - 1
  nr_frames = nr_frames_after
  nr_neurons = activity.shape[0]

  trial_matrix_3d = np.zeros( (nr_trials, nr_neurons, nr_frames))
  image_index = np.zeros( nr_trials )
  is_change = np.zeros(nr_trials)

  for i in range(nr_trials):
    stim_row = stimulus_details.iloc[i+1]  # skip first entry
    start_time = stim_row.start_time

    start_idx = np.argmin( np.abs( neuron_time - start_time))
    trial_matrix_3d[i,:,:] = activity[:,start_idx:start_idx+nr_frames]

    image_index[i] = stim_row.image_index
    is_change[i] = stim_row.is_change

  return trial_matrix_3d, image_index, is_change
