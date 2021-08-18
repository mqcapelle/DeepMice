# Imports
# Standard library imports
from pathlib import Path
import time
# Third party imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# Local library imports
from DeepMice.models.GRU import GRU
from DeepMice.utils.train_and_test import train, test
from DeepMice.utils.sequential_data_loader import get_train_val_test_loader
from DeepMice.utils.helpers import set_device


if __name__ == "__main__":
    # ################################################
    # Settings
    # ################################################
    # Paths
    path_relative = Path('/Users/mc/PycharmProjects/DeepMice/')
    path_to_data_folder = path_relative.joinpath('DeepMice/data/')
    path_to_state_dict_folder = path_relative.joinpath('DeepMice/state_dicts')
    path_to_figure_folder = path_relative.joinpath('DeepMice/figures')

    # Session numbers; please enter a three digit string
    train_session_number = '006'
    transfer_session_number = '040'

    # Training settings
    SEED = 24872
    learn_rate = 0.0009220919769660428  # 1e-3
    patience = 5
    n_epochs_train = 50
    n_epochs_transfer = 10

    # ################################################
    # Initialise
    # ################################################
    DEVICE = set_device()
    path_to_state_dict_train = path_to_state_dict_folder.joinpath(f'GRU_train_{train_session_number}.pt')
    path_to_state_dict_transfer = path_to_state_dict_folder.joinpath(f'GRU_transfer_{transfer_session_number}.pt')
    path_to_session_file_train = list(path_to_data_folder.glob(f'{train_session_number}_*.nc'))[0]
    path_to_session_file_transfer = list(path_to_data_folder.glob(f'{transfer_session_number}_*.nc'))[0]

    # ################################################
    # Initial training
    # ################################################
    # Load data
    train_loader, val_loader, test_loader = get_train_val_test_loader(path_to_session_file_train, SEED=SEED)
    X_b, y_b, init_b = next(iter(train_loader))
    train_neurons = X_b.size(2)

    print(f"--- INITIAL TRAINING ---")
    print(f"Number of neurons {train_neurons}")

    # Initialize model
    train_model = GRU(example_batch_X=X_b)

    if not path_to_state_dict_train.exists():
        # Train and validate model
        best_train_model, best_epoch_train,\
        train_loss_train, train_acc_train,\
        validation_loss_train, validation_acc_train = train(
            model=train_model,
            train_loader=train_loader,
            valid_loader=val_loader,
            device=DEVICE,
            learn_rate=learn_rate, n_epochs=n_epochs_train, patience=patience,
            freeze=False,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam
        )

        # Save model state_dict
        torch.save(best_train_model.state_dict(), path_to_state_dict_train)
    else:
        # Load trained model parameters to new model (except for encoder layer)
        loaded_state_dict = torch.load(path_to_state_dict_train)
        own_state_dict = train_model.state_dict()
        for name, param in loaded_state_dict.items():
            if name in own_state_dict.keys():
                param = param.data
                own_state_dict[name].copy_(param)

    # Test model
    accuracy_train = test(model=best_train_model, test_loader=test_loader, device=DEVICE)

    # Print results
    print(f"Train model | Test acc {accuracy_train*100:3.1f} % | Neurons: {train_neurons} | Session: {train_session_number}")

    # ################################################
    # Transfer training
    # ################################################
    # Load data
    train_loader, val_loader, test_loader = get_train_val_test_loader(path_to_session_file_transfer, SEED=SEED)
    X_b, y_b, init_b = next(iter(train_loader))
    transfer_neurons = X_b.size(2)

    print(f"--- TRANSFER LEARNING ---")
    print(f"Number of neurons {train_neurons}")

    if not path_to_state_dict_transfer.exists():
        # Initialize model
        transfer_model = GRU(example_batch_X=X_b)

        # Train and validate model
        best_transfer_model, best_epoch_transfer,\
        train_loss_transfer, train_acc_transfer,\
        validation_loss_transfer, validation_acc_transfer = train(
            model=transfer_model,
            train_loader=train_loader,
            valid_loader=val_loader,
            device=DEVICE,
            learn_rate=learn_rate, n_epochs=n_epochs_transfer, patience=patience,
            freeze=True, path_to_model_state_dict=path_to_state_dict_train,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam
        )

        # Test model
        accuracy_transfer = test(model=best_transfer_model, test_loader=test_loader, device=DEVICE)

        # Print results
        print(f"Transfer model | Test acc {accuracy_transfer * 100:3.1f} % | Neurons: {transfer_neurons} | Session: {transfer_session_number}")

        # Save model state_dict
        torch.save(best_transfer_model.state_dict(), path_to_state_dict_transfer)


fig, ax1 = plt.subplots(1, 1)
ax1.set_title(f"Train {train_neurons} neurons ({train_session_number}), Transfer {transfer_neurons} neurons ({transfer_session_number})")
ax1.plot(train_acc_train, 'tab:blue', linestyle='dotted', label='train (train)')
ax1.plot(train_acc_transfer,  'tab:red', linestyle='dotted', label='transfer (train)')
ax1.plot(validation_acc_train, 'tab:blue', linestyle='solid',  label='train (validation')
ax1.plot(validation_acc_transfer,  'tab:red', linestyle='solid', label='transfer (validation)')
ax1.set_ylim([0.5, 1])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()

plt.savefig(path_to_figure_folder.joinpath(f'train_{train_neurons:3.0f}neurons_{train_session_number}session_Transfer_{transfer_neurons:3.0f}neurons_{train_session_number}session'))

