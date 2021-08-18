import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
# Local library imports
from DeepMice.utils.train_and_test import train, test
from DeepMice.utils.sequential_data_loader import get_train_val_test_loader
from DeepMice.utils.helpers import set_device


class GRU(nn.Module):
    def __init__(self, example_batch_X, x_feat_length=69, gru_hidden_size=20):
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

        return y_logit


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
    X_b, train_loader, val_loader, test_loader = get_train_val_test_loader(path_to_session_file_train)

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
    X_b, train_loader, val_loader, test_loader = get_train_val_test_loader(path_to_session_file_transfer)

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

