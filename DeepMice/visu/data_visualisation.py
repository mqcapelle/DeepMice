

from pathlib import Path

import matplotlib.pyplot as plt

from DeepMice.obsolete.data_loader import load_example_data


def plot_traces_above(time, traces, axis=None, key=None, spacing=0.5, linewidth=0.5, alpha=1):
    """Plot calcium traces on top of each other with white area below
    Adrian 2020-08-03 """

    if axis is None:
        plt.figure(figsize=(9, 7))
    else:
        plt.sca(axis)

    for i, trace in enumerate( traces ):

        plt.plot(time, trace + spacing*i, lw=linewidth, zorder=500-i, color='C0')
        plt.fill_between(time, trace + spacing*i, 0, color = 'white', zorder=500-i, alpha=alpha)

    plt.xlabel('Time [s]')
    plt.xlim(left=-1)
    plt.ylim(bottom=-1)

    return axis

# Paths
path_to_data_folder = Path("DeepMice/data")
path_to_data_file = path_to_data_folder.joinpath("one_example_session.npy")

# Load data
data = load_example_data(path_to_data_file, print_keys=True)

neuron_selection = 5

time = data["neuron_time"]
# Plot activity over time
fluoresence = data['neuron_fluoresence']
activity = data['neuron_activity']
neuron_time = data['neuron_time']

example_neuron = 10
x_start = 300
x_end = 400



fig, axes = plt.subplots(nrows=1, ncols=2, tight_layout=True)
time_slice = (neuron_time > x_start) & (neuron_time < x_end)
plot_traces_above(neuron_time[time_slice], activity[0:100, time_slice],
                  axis=axes[0], spacing=0.2)
axes[0].set_xlim((x_start, x_end))
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].set_yticks([])

# axes[1].plot(neuron_time, fluoresence[example_neuron], label=r"$\Delta$F/F")
axes[1].plot(neuron_time, activity[example_neuron], label="activity")
axes[1].set_xlabel('Time [seconds]')
axes[1].set_ylabel('Activity [a.u.]')
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].set_xlim((x_start, x_end))
# axes[1].legend(frameon=False)

plt.show()


time_change = data["stim_details_pandas"].query("is_change")["start_time"]
time_licks = data["licks_pandas"]["timestamps"]
time_rewards = data["reward_pandas"]["timestamps"]
time_spikes = (time * (activity > 0.01))[:, 45000:]

# Behavior, stimulus and activity spikes over time
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, tight_layout=True)
ax1.eventplot(time_spikes)
ax2.eventplot([time_change, time_licks, time_rewards], colors=["tab:blue", "tab:orange", "tab:red"])
ax1.set_xlim(left=4200)
plt.show()

