import copy
import time

import torch
import torch.nn as nn
# Local library imports


def train(model, train_loader, valid_loader, device,
          learn_rate, n_epochs, patience,
          freeze=False, path_to_model_state_dict=None,
          warm_up=0, reduce_epoch=0.2,
          criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):

    time_start = time.time_ns()

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

    for epoch in range(n_epochs):
        # Train
        model.train()  # Tell model that we start training

        # Reset loss and accuracy
        running_loss = 0
        running_correct = 0
        running_total = 0

        for batch_index, batch in enumerate(train_loader):
            if reduce_epoch < 1:
                # only use a fraction of the epochs to reduce the effective
                # number of batches per epoch
                if np.random.rand() > reduce_epoch:
                    continue  # skip this batch
                    
            # send to GPU
            X_batch, y_batch, init_batch = batch
            y_batch = y_batch.long()   # change datatype of prediction to long int
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Optimize model
            optimizer.zero_grad()
            output = model(X_batch)
            # output shape (nr_batches, len_sequence, n_images)

            if warm_up > 0:
                # remove the first few images
                output = output[:,warm_up:, :]
                y_batch = y_batch[:,warm_up:, :]
            
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
            
            if warm_up > 0:
                # remove the first few images
                output = output[:,warm_up:, :]
                y_batch = y_batch[:,warm_up:, :]
            
            loss = criterion(torch.transpose(output, 1, 2), y_batch)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, dim=2)
            val_correct += (predicted == y_batch).sum().to('cpu').numpy()  # Simply copied .sum(0).numpy() from RNN.py
            val_total += torch.numel(y_batch)

        validation_loss.append(val_loss / len(valid_loader))
        validation_acc.append(val_correct / val_total)

        print(f"Epoch {epoch:2.0f} | Train acc: {train_acc[-1] * 100:3.2f} | Val acc {validation_acc[-1] * 100:3.2f} | "
              f"Train loss: {train_loss[-1]:1.3f} | Val loss: {validation_loss[-1]:1.3f} | Time passed: {(time.time_ns() - time_start) * 1E-9:3.2f} s")

        # Early stopping
        if (validation_acc[-1] > best_acc):
            best_acc = validation_acc[-1]
            best_epoch = epoch
            best_model = copy.deepcopy(model).to('cpu')
            wait = 0
        else:
            wait += 1

        if wait > patience:
            print(f'Early stopped. Best epoch {best_epoch} | Val accuracy {validation_acc[best_epoch] * 100:3.2f} | '
                  f'Time passed: {(time.time_ns() - time_start) * 1E-9:4.2f} s')
            break

    return best_model, best_epoch, train_loss, train_acc, validation_loss, validation_acc


def test(model, device, test_loader, warm_up=0, return_results=False):
    model.eval()
    model.to(device)
    
    if return_results:
        true_image = []
        pred_image = []
    test_correct, test_total = 0, 0
    
    for batch_index, batch in enumerate(test_loader):
        # send to GPU
        X_batch, y_batch, init_batch = batch
        y_batch = y_batch.long()  # change datatype of prediction to long int
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        output = model(X_batch)
        if warm_up > 0:
            # remove the first few images
            output = output[:,warm_up:, :]
            y_batch = y_batch[:,warm_up:, :]
          
        # Calculate accuracy
        _, predicted = torch.max(output, dim=2)
        test_correct += (predicted == y_batch).sum().to('cpu').numpy()  # Simply copied .sum(0).numpy() from RNN.py
        test_total += torch.numel(y_batch)

        if return_results:
            true_image.append( y_batch.to('cpu').numpy() )
            pred_image.append( predicted.to('cpu').numpy() )
        
    accuracy = test_correct / test_total
    
    if return_results == False:
        return accuracy
    else:
        return accuracy, true_image, pred_image
