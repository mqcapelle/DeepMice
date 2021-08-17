import torch
import torch.nn as nn
import torch.nn.functional as F


def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("DEVICE is set to CPU")
  else:
    print("DEVICE is set to GPU")

  return device

DEVICE = set_device()


class Net(nn.Module):
    def __init__(self, n_neurons, n_time):
        super().__init__()
        # Assign parameters
        self.n_neurons = n_neurons
        self.n_time = n_time

        # Define layers
        self.conv1 = nn.Conv1d(
            in_channels=self.n_neurons, out_channels=10,
            kernel_size=2, device=DEVICE,
        )
        self.fc1 = nn.Linear(
            in_features=10, out_features=100,
            bias=True, device=DEVICE,
        )
        self.fc2 = nn.Linear(
            in_features=100, out_features=1,
            bias=True, device=DEVICE,
        )

    def forward(self, x):
        # Convolutional layer
        x = F.relu(self.conv1(x))
        # Linear layer
        x = F.relu(self.fc1(x))
        # Linear layer
        x = F.relu(self.fc2(x))
        return x











