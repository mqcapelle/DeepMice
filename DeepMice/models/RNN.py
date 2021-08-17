import os
import numpy as np
import torch
import torch.nn as nn

from DeepMice.utils.data_loader_class import DeepMiceDataLoader
# from DeepMice.utils.data_loader import load_one_session, easy_train_test_loader


class LSTM(nn.Module):
  def __init__(self, input_size, embed_size, n_hidden, hidden_size,
               output_size, device, bi=True):
    '''Define a LSTM model

    Note that for our task, the data corresponds to:
    N = batch size = sessions
    L = sequence lenght = time points
    Hin = input size = neurons

    input_size: number of input units, should correspond to Hin
    embed_size: number of output units from the first layer, arbitrary
    n_hidden: number of hidden layers of the RNN
    hidden_size: number of hidden units for each hidden layer of the RNN
    output_size: number of readout units, should correspond to our features
    device: 'cpu' or 'cuda'
    bi: if the LSTM will be bidirectional, default = True
    '''
    super(LSTM, self).__init__()
    self.input_size = input_size
    self.embed_size = embed_size
    self.n_hidden = n_hidden
    self.hidden_size = hidden_size
    self.output_size = output_size
    # self.neurons_size = neurons_size
    self.device = device
    if bi is True:
      self.D = 2
    elif bi is False:
      self.D = 1

    self.fc1 = nn.Linear(input_size, embed_size, bias=True)

    self.relu = nn.ReLU()

    self.w_ie = torch.normal(1., 1/np.sqrt(input_size * embed_size),
                             size=(embed_size, input_size))
    with torch.no_grad():
      self.fc1.weight = nn.Parameter(self.w_ie)
    # Define the word embeddings
    # self.word_embeddings = nn.Embedding(neurons_size, embed_size)
    # Define the dropout layer
    # self.dropout = nn.Dropout(0.5)
    # Define the bilstm layer
    self.bilstm = nn.LSTM(embed_size, hidden_size,
                          num_layers=self.n_hidden,
                          batch_first=True,
                          bidirectional=bi)
    # Define the fully-connected layer
    self.fc2 = nn.Linear(self.D * hidden_size, output_size)


  def forward(self, input_seqence):
    # N = batch size (sessions)
    # L = sequence lenght (time points)
    # Hin = input size (neurons)

    # Data trough the first input layer should be of dims [N, ..., Hin]
    fc1_in = self.fc1(input_seqence)
    # print(fc1_in.size())
    # Rectified Linear Units activation
    fc1_out = self.relu(fc1_in)
    # print(fc1_out.size())

    # Define hidden layers of the RNN
    hidden = (torch.randn(self.n_hidden * self.D, fc1_in.size()[0],
                          self.hidden_size).to(self.device),
              torch.randn(self.n_hidden * self.D, fc1_in.size()[0],
                          self.hidden_size).to(self.device))
    # RNN activation
    # Data trough the RNN should be of dim [N, L, Hin]
    recurrent, hidden = self.bilstm(fc1_out, hidden)
    rec_out = self.relu(recurrent)

    # Pass data in the last full connected layer and apply softmax
    fc2_in = self.fc2(rec_out)
    fc2_out = torch.softmax(fc2_in, dim=-1).permute(0, 2, 1)

    return fc2_out


# Training function
def train(model, device, train_iter, valid_iter, epochs, learning_rate,
          criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam):
  criterion = criterion()
  optimizer = optimizer(model.parameters(), lr=learning_rate)

  train_loss, validation_loss = [], []
  train_acc, validation_acc = [], []

  for epoch in range(epochs):
    #train
    model.train()
    running_loss = 0.
    correct, total = 0, 0
    steps = 0

    for idx, batch in enumerate(train_iter):
      # text = batch.text[0]
      data = torch.swapaxes(batch[0], -1, 1).float()
      # print(data.size())
      # print(type(text), text.shape)
      target = torch.squeeze(batch[1], -1).expand(-1, data.size(1)).float()
      # print(target.size())
      target = torch.autograd.Variable(target).long()
      data, target = data.to(device), target.to(device)

      # add micro for coding training loop
      optimizer.zero_grad()
      output = model(data)
      # print(output)
      # print(output.size())

      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      steps += 1
      running_loss += loss.item()

      # get accuracy
      _, predicted = torch.max(output, 1)
      # print(predicted, predicted.size(), target.size())
      total += target.size(0)
      # print(total)
      correct += (predicted == target).sum(0).numpy()

    train_loss.append(running_loss/len(train_iter))
    train_acc.append(correct/total)

    print('Epoch {0}\n'.format(epoch + 1),
          'Training loss: {0}\n'.format(running_loss/len(train_iter)),
          'Training accuracy: {0}'.format(100*correct/total))

    # print(f'Epoch: {epoch + 1}, '
    #       f'Training Loss: {running_loss/len(train_iter):.4f}, '
    #       f'Training Accuracy: {100*correct/total: .2f}%')

    # evaluate on validation data
    model.eval()
    running_loss = 0.
    correct, total = 0, 0

    with torch.no_grad():
      for idx, batch in enumerate(valid_iter):
        # text = batch.text[0]
        data = torch.swapaxes(batch[0], -1, 1).float()
        target = torch.squeeze(batch[1], -1).expand(-1, data.size(1)).float()
        target = torch.autograd.Variable(target).long()
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        running_loss += loss.item()

        # get accuracy
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum(0).numpy()

    validation_loss.append(running_loss/len(valid_iter))
    validation_acc.append(correct/total)

    print('Epoch {0}\n'.format(epoch + 1),
          'Validation Loss: {0}\n'.format(running_loss/len(valid_iter)),
          'Validation Accuracy: {0}'.format(100*correct/total))
    # print(f'Validation Loss: {running_loss/len(valid_iter):.4f}, '
    #       f'Validation Accuracy: {100*correct/total: .2f}%')

  return model, train_loss, train_acc, validation_loss, validation_acc


# Testing function
def test(model, device, test_iter):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for idx, batch in enumerate(test_iter):
      data = torch.swapaxes(batch[0], -1, 1).float()
      target = torch.squeeze(batch[1], -1).expand(-1, 21).float()
      # print(target.size())
      target = torch.autograd.Variable(target).long()
      data, target = data.to(device), target.to(device)

      outputs = model(data)
      _, predicted = torch.max(outputs, 1)
      total += target.size(0)
      correct += (predicted == target).sum(0).numpy()

    acc = 100 * correct / total
    return acc

if __name__ == '__main__':
  # example use of the data loader (assumes file to be in working directory)

  path = '/mnt/sda5/python_projects/public/DeepMice/018_excSession_v1_ophys_856295914.nc'
  if not os.path.isfile(path):
    raise Exception(
      'Example file "{}" is not in working directory.'.format(path))

  my_class = DeepMiceDataLoader(path)
  my_class.setup()

  train_trials = my_class.train_loader
  validate_trials = my_class.val_loader
  test_trials = my_class.test_loader

  X_batch, y_batch = next(iter(train_trials))
  insz = X_batch.shape[1]
  ousz = len(np.unique(y_batch))
  # print(insz, ousz)

  model = LSTM(input_size=insz, embed_size=200, n_hidden=2, hidden_size=100,
               output_size=ousz, device='cpu', bi=False)

  trained_model, train_loss, train_acc, validation_loss, validation_acc = \
    train(model, 'cpu', train_trials, validate_trials, 50, 0.005,
        criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam)

  test_accuracy = test(trained_model, 'cpu', test_trials)
  print('\nThe test accuracy is: {0}'.format(test_accuracy))
