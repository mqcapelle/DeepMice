
import copy
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Local library imports
from DeepMice.utils.sequential_data_loader import DataInterface, SequentialDataset
from DeepMice.utils.helpers import set_seed, set_device, seed_worker


class GRU(nn.Module):
    def __init__(self, example_batch_X, x_feat_length=20, gru_hidden_size=9):
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
          learn_rate, n_epochs, patience,
          freeze=False, path_to_model_state_dict=None,
          criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):

    model.to(device)
    # Assign optimizer
    if freeze:
        # Set only encoder_layer to be optimised
        optimizer = optimizer(model.encoder_layer.parameters(), lr=learn_rate)
        # Load trained model parameters to new model (except for encoder layer)
        trained_state_dict = torch.load(path_to_model_state_dict)
        transfer_state_dict = model.state_dict()
        for name, param in trained_state_dict.items():
            if "encoder" not in name:
                param = param.data
                transfer_state_dict[name].copy_(param)
    else:
        # Set all layers to be optimised
        optimizer = optimizer(model.parameters(), lr=learn_rate)

    # Allocate loss and accuracy data
    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []
    best_acc = -1e3  # below first accuracy for sure...
    wait = 0

    # early_stopping = EarlyStopping(patience=patience, verbose=True)
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

            running_correct += (predicted == y_batch).sum().to('cpu').numpy()  # Simply copied .sum(0).numpy() from RNN.py
            running_total += torch.numel(y_batch)

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(running_correct / running_total)

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
            val_correct += (predicted == y_batch).sum().to('cpu').numpy()  # Simply copied .sum(0).numpy() from RNN.py
            val_total += torch.numel(y_batch)

        validation_loss.append(val_loss / len(valid_loader))
        validation_acc.append(val_correct / val_total)

        print(f"Epoch {epoch:3.0f} | Training loss:     {train_loss[-1]:1.3f} | Validation loss:     {validation_loss[-1]:1.3f}\n"
              f"            Training accuracy: {train_acc[-1] * 100:3.2f}  | Validation accuracy: {validation_acc[-1] * 100:3.2f}"
              )

        # Early stopping
        if (validation_acc[-1] > best_acc):
            best_acc = validation_acc[-1]
            best_epoch = epoch
            best_model = copy.deepcopy(model).to('cpu')
            wait = 0
        else:
            wait += 1

        if wait > patience:
            print(f'Early stopped with best epoch: {best_epoch}')
            break

    return best_model, best_epoch, train_loss, train_acc, validation_loss, validation_acc


def test(model, device, test_loader):
    model.eval()
    model.to(device)

    test_correct, test_total = 0, 0
    for batch_index, batch in enumerate(test_loader):
        # send to GPU
        X_batch, y_batch, init_batch = batch
        y_batch = y_batch.long()  # change datatype of prediction to long int
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        output = model(X_batch)

        # Calculate accuracy
        _, predicted = torch.max(output, dim=2)
        test_correct += (predicted == y_batch).sum().to('cpu').numpy()  # Simply copied .sum(0).numpy() from RNN.py
        test_total += torch.numel(y_batch)

    accuracy = test_correct / test_total
    return accuracy


def quick_and_dirty_dataloader(path_to_session_file):
    data_interface = DataInterface(path=path_to_session_file)
    dataset_train = SequentialDataset(data_interface, part='train', len_sequence=15)
    dataset_val = SequentialDataset(data_interface, part='val', len_sequence=15,)
    dataset_test = SequentialDataset(data_interface, part='test', len_sequence=15,)

    # set seeds
    SEED = 2021
    g_seed = torch.Generator()
    g_seed.manual_seed(SEED)
    set_seed(seed=SEED)

    train_loader = DataLoader(dataset_train, batch_size=5, shuffle=True,
                              worker_init_fn=seed_worker, generator=g_seed)

    val_loader = DataLoader(dataset_val, batch_size=5, shuffle=False,
                              worker_init_fn=seed_worker, generator=g_seed)

    test_loader = DataLoader(dataset_test, batch_size=5, shuffle=False,
                              worker_init_fn=seed_worker, generator=g_seed)

    X_b, y_b, init_b = next(iter(train_loader))
    # print(X_b.shape,    # Neural activity (batch_size, len_sequence, nr_neurons, time)
    #       y_b.shape,    # Shown images    (batch_size, len_sequence)
    #       init_b.shape) # Initial image   (batch_size, 1)
    return X_b, train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Paths
    session_number = 20
    path_to_data_folder = Path('/Users/mc/PycharmProjects/DeepMice/DeepMice/data/')
    path_to_session_file_train = list(path_to_data_folder.glob(f'*{session_number}_*.nc'))[0]
    path_to_session_file_transfer = list(path_to_data_folder.glob(f'*030_*.nc'))[0]

    now = datetime.datetime.now().strftime("%H%M%S")
    path_to_model_state_dict = Path(f'/Users/mc/PycharmProjects/DeepMice/DeepMice/models/GRU.pt')

    DEVICE = set_device()
    # ################################################
    # Initial training
    # ################################################
    # Load data
    X_b, train_loader, val_loader, test_loader = quick_and_dirty_dataloader(path_to_session_file_train)

    # Initialize model
    model = GRU(example_batch_X=X_b)

    # Train and validate model
    train_model, best_epoch, train_loss, train_acc, validation_loss, validation_acc = train(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        device=DEVICE,
        learn_rate=1e-3, n_epochs=1, patience=5,
        freeze=False,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam
    )

    # Test model
    accuracy = test(model=train_model,
                    test_loader=test_loader,
                    device=DEVICE)

    # Save the model parameters
    torch.save(train_model.state_dict(), path_to_model_state_dict)

    # ################################################
    # Transfer training
    # ################################################
    # Load data
    X_b, train_loader, val_loader, test_loader = quick_and_dirty_dataloader(path_to_session_file_transfer)

    # Initialize model
    transfer_model = GRU(example_batch_X=X_b)

    # Train and validate model
    best_transfer_model, best_epoch, train_loss, train_acc, validation_loss, validation_acc = train(
        model=transfer_model,
        train_loader=train_loader,
        valid_loader=val_loader,
        device=DEVICE,
        learn_rate=1e-3, n_epochs=2, patience=5,
        freeze=True, path_to_model_state_dict=path_to_model_state_dict,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam
    )

