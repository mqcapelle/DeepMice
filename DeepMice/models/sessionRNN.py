
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm

class SessionRNN(nn.Module):
    def __init__(self, n_neurons=1000, n_time=7, encdec_hidden_size=20, rnn_hidden_size=10, rnn_layers=2):
        super().__init__()

        # Assign parameters
        self.n_neurons = n_neurons
        self.n_time = n_time
        self.encdec_hidden_size = encdec_hidden_size  # length of  encoding/decoding hidden layer
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers

        # Define layers
        # # Encoding layer
        self.enc_lin = nn.Linear(n_time, self.encdec_hidden_size)
        # # Recurrent layer
        self.rnn = nn.RNN(self.encdec_hidden_size, self.rnn_hidden_size, self.rnn_layers)
        # # Decoding layer
        self.dec_lin = nn.Linear(self.rnn_hidden_size * self.n_neurons, 8)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        # Encoding layer
        encoded = F.relu(self.enc_lin(input))
        # TODO: add dropout layer
        # Recurrent layer
        output, hidden = self.rnn(encoded.reshape(1, batch_size, -1), hidden)
        #Flattening the rnn output
        output = torch.flatten(output,1)

        print(output.shape)
        # Decoding
        output = self.dec_lin(output.reshape(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size),
                    torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size))

        return torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size)


def test(model, data_loader):

    correct = 0
    total = 0
    for data in data_loader:
      inputs, labels = data
      inputs = inputs.to(device).float()
      labels = labels.to(device).long()

      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return total, acc


def train(model, train_loader, criterion=nn.CrossEntropyLoss(), lr=3e-4, num_epochs=50 ):
    optimizer = optim.Adam(model.parameters(), lr)

    model.to(device)
    model.train()  # Tell the model that we are training it

    training_losses = []
    for epoch in tqdm(range(num_epochs)):
        inputs, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        print(f"Input Shape:{inputs.shape}")
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        if verbose:
            training_losses += [loss.item()]

    #Saving the model, we do this in the main function
    #x=inputs[0,:,:]
    #traced_cell = torch.jit.trace(model, (x))
    #torch.jit.save(traced_cell, "model.pth")

    return model

