
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Local library imports
# from DeepMice.utils.easy_load import DataHandler
from DeepMice.utils.sequential_data_loader import DataInterface, SequentialDataset


class GRU(nn.Module):
    def __init__(self, example_batch_X, x_feat_length=20, gru_hidden_size=9,):
        super().__init__()
        # Assign parameters
        self.n_batches = example_batch_X.shape[0]  # However, we do not need this
        self.n_trials = example_batch_X.shape[1]
        self.n_neurons = example_batch_X.shape[2]
        self.n_timestamps = example_batch_X.shape[3]
        self.x_feat_length = x_feat_length
        self.gru_hidden_size = gru_hidden_size
        self.n_images = 8 + 1


        # Define layers
        self.encoder_layer = nn.Linear(in_features=self.n_neurons*self.n_timestamps,
                             out_features=self.x_feat_length,
                             bias=True)
        self.rnn = nn.GRU(input_size=self.x_feat_length,
                          hidden_size=self.gru_hidden_size,
                          num_layers=1,
                          batch_first=True)
        self.decoder_layer = nn.Linear(in_features=self.gru_hidden_size,
                                       out_features=self.n_images,
                                       bias=True)

    def forward(self, input_batch, ):
        """ Forward pass through the network

        input_batch: (nr_batches, len_sequence, nr_neurons, time)
        """
        # flatten neurons, time dimension
        x = torch.flatten(input_batch, start_dim=2, end_dim=-1)
        # x shape: (nr_batches, len_sequence, nr_neurons*time)

        # Encoding
        x = self.encoder_layer(x)
        x = F.relu(x)
        # x shape: (nr_batches, len_sequence, x_feat_length)

        # Recurrent network
        x, h_n = self.rnn(x, )  # h_0 can still be given
        # x shape: (nr_batches, len_sequence, gru_hidden_size)

        # Decoding
        y_logit = self.decoder_layer(x)
        # y_logit shape: (nr_batches, len_sequence, n_images)

        # Softmax
        # y_pred = torch.softmax(x, )
        # y shape: (nr_batches, len_sequence, 1)

        return y_logit


def train(model, train_loader, valid_loader, device,
          learn_rate, n_epochs, criterion=nn.CrossEntropyLoss(),
          optimizer=torch.optim.Adam):
    # Assign optimizer parameters
    optimizer = optimizer(model.parameters(), lr=learn_rate)

    # Allocate loss and accuracy data
    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []

    for epoch in range(n_epochs):
        # Train
        model.train()  # Tell model that we start training

        # Reset loss and accuracy
        running_loss = 0
        running_correct = 0
        running_total = 0

        for batch_index, batch in enumerate(train_loader):
            # send to GPU
            X_batch, y_batch, init_batch = batch
            y_batch = y_batch.long()   # change datatype of prediction to long int
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Optimize model
            optimizer.zero_grad()
            output = model(X_batch)
            # output shape (nr_batches, len_sequence, n_images)

            # Calculate loss and optimize
            loss = criterion(torch.transpose(output, 1, 2), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, dim=2)

            running_correct += (predicted == y_batch).sum().numpy()  # Simply copied .sum(0).numpy() from RNN.py
            running_total += torch.numel(y_batch)

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(running_correct / running_total)

        if epoch % 2 == 0:

            # Evaluate on validation data
            model.eval()
            val_loss = 0.
            val_correct, val_total = 0, 0

            for batch_index, batch in enumerate(valid_loader):
                # send to GPU
                X_batch, y_batch, init_batch = batch
                y_batch = y_batch.long()  # change datatype of prediction to long int
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                output = model(X_batch)
                # output shape (nr_batches, len_sequence, n_images)

                loss = criterion(torch.transpose(output, 1, 2), y_batch)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(output, dim=2)
                val_correct += (predicted == y_batch).sum().numpy()  # Simply copied .sum(0).numpy() from RNN.py
                val_total += torch.numel(y_batch)

            validation_loss.append(val_loss / len(valid_loader))
            validation_acc.append(val_correct / val_total)

            print(f"Epoch {epoch}: \n"
                  f"   Training loss:     {train_loss[-1]:1.3f}  | Validation loss:          {validation_loss[-1]:1.3f}\n"
                  f"   Training accuracy: {train_acc[-1] * 100:3.2f}  | Validation accuracy: {validation_acc[-1] * 100:3.2f}"
                  )


def test(model, device, test_loader):
    model.eval()

    test_correct, test_total = 0, 0
    for batch_index, batch in enumerate(test_loader):
        # send to GPU
        X_batch, y_batch, init_batch = batch
        y_batch = y_batch.long()  # change datatype of prediction to long int
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        output = model(X_batch)

        # Calculate accuracy
        _, predicted = torch.max(output, dim=2)
        test_correct += (predicted == y_batch).sum().numpy()  # Simply copied .sum(0).numpy() from RNN.py
        test_total += torch.numel(y_batch)

    accuracy = test_correct / test_total
    return accuracy


if __name__ == "__main__":
    # Paths
    # ex_path = Path('/Users/mc/PycharmProjects/DeepMice/DeepMice/data/020_excSession_v1_ophys_858863712.nc')    # example file
    ex_path = Path('/Users/mc/PycharmProjects/DeepMice/DeepMice/data/018_excSession_v1_ophys_856295914.nc')

    # Load data
    data_interface = DataInterface(path=ex_path)
    dataset_train = SequentialDataset(data_interface,
                                part='train',    # or 'test', 'val'
                                len_sequence=15
                                )

    train_loader = DataLoader(dataset_train,
                              batch_size=5,
                              shuffle=True,
                              worker_init_fn=3453)

    dataset_val = SequentialDataset(data_interface,
                                    part='val',  # or 'test', 'val'
                                    len_sequence=15,)

    val_loader = DataLoader(dataset_val,
                            batch_size=5,
                            shuffle=False,
                            worker_init_fn=3453)

    dataset_test = SequentialDataset(data_interface,
                                     part='test',  # or 'test', 'val'
                                     len_sequence=15,)

    test_loader = DataLoader(dataset_test,
                             batch_size=5,
                             shuffle=False,
                             worker_init_fn=3453)

    X_b, y_b, init_b = next( iter(train_loader) )

    # print(X_b.shape,    # Neural activity (batch_size, len_sequence, nr_neurons, time)
    #       y_b.shape,    # Shown images    (batch_size, len_sequence)
    #       init_b.shape) # Initial image   (batch_size, 1)

    # Initialize model
    model = GRU(example_batch_X=X_b)

    # Train and validate
    train(model=model,
          train_loader=train_loader,
          valid_loader=val_loader,
          device='cpu',
          learn_rate=1e-3, n_epochs=50,
          criterion=nn.CrossEntropyLoss(),
          optimizer=torch.optim.Adam)

    # Test
    accuracy = test(model=model,
                    test_loader=test_loader,
                    device='cpu')
