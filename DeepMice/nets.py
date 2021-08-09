import torch
import torch.nn as nn

class MLPNet(nn.Module):
# from https://deeplearning.neuromatch.io/tutorials/W1D3_MultiLayerPerceptrons/student/W1D3_Tutorial2.html
  def __init__(self, actv, input_feature_num, hidden_unit_nums, output_feature_num):
    super(MLPNet, self).__init__()
    self.input_feature_num = input_feature_num # save the input size for reshapinng later
    self.mlp = nn.Sequential() # Initialize layers of MLP

    in_num = input_feature_num # initialize the temporary input feature to each layer
    for i in range(len(hidden_unit_nums)): # Loop over layers and create each one
      out_num = hidden_unit_nums[i] # assign the current layer hidden unit from list
      layer = nn.Linear(in_num, out_num) # use nn.Linear to define the layer
      in_num = out_num # assign next layer input using current layer output
      self.mlp.add_module(f"Linear_{i}", layer) # append layer to the model with a name

      actv_layer = eval(f"nn.{actv}") # Assign activation function (eval allows us to instantiate object from string)
      self.mlp.add_module(f"Activation_{i}", actv_layer) # append activation to the model with a name

    out_layer = nn.Linear(in_num, output_feature_num) # Create final layer
    self.mlp.add_module('Output_Linear', out_layer) # append the final layer

  def forward(self, x):
    # reshape inputs to (batch_size, input_feature_num)
    # just in case the input vector is not 2D, like an image!
    x = x.view(-1, self.input_feature_num)

    logits = self.mlp(x) # forward pass of MLP
    return logits
