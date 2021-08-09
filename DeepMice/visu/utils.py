import os
import os.path as op
import numpy as np
import xarray as xr


def events_to_timeseries(events, precision=2):
    events = np.array(events)
    events = events.round(precision)
    _eve = events * (10 ** precision)
    min_len = int(_eve.min())
    max_len = int(_eve.max() + 1)
    pos = _eve.astype(int)
    ts = np.zeros(max_len)
    ts[pos] += 1
    ts = ts[min_len:]
    tp = np.arange(pos[0], pos[-1] + 1, 1) / (10 ** precision)
    return ts, tp


def get_time_series(file):

    if isinstance(file, str):
        data = np.load(file, allow_pickle=True).item()
    elif isinstance(file, dict):
        data = file
    print('Keys of data dictionary:\n', data.keys())

    neu_data = xr.DataArray(data['neuron_activity'],
                            coords=[list(range(data['neuron_activity'].shape[0])),
                                    data['neuron_time']],
                            dims=['neuron', 'times'],
                            name='neurons')

    speed_data = xr.DataArray(data['running_pandas']['speed'],
                              coords=[data['running_pandas']['timestamps']],
                              dims=['times'],
                              name='speed')

    lick_data = events_to_timeseries(data['licks_pandas']['timestamps'])
    lick_data = xr.DataArray(lick_data[0], coords=[lick_data[1]],
                             dims=['times'], name='licking')

    reward_data = events_to_timeseries(data['reward_pandas']['timestamps'])
    reward_data = xr.DataArray(reward_data[0], coords=[reward_data[1]],
                               dims=['times'], name='reward')

    neu_data = neu_data.interp(times=speed_data.times, method='slinear')
    lick_data = lick_data.interp(times=speed_data.times, method='zero')
    reward_data = reward_data.interp(times=speed_data.times, method='zero')

    # neu_data = neu_data.assign_coords({'speed': ('times', speed_data.values),
    #                                    'licks': ('times', lick_data.values),
    #                                    'reward': ('times', reward_data.values)}
    #                                   )

    neu_data = xr.merge([neu_data, speed_data, lick_data, reward_data])

    return neu_data

# if __name__ == '__main__':
#     dir = os.getcwd()
#     filename = op.join(dir, 'one_example_session.npy')
#     get_time_series(filename)
