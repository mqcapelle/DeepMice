from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN

from DeepMice.utils.sequential_data_loader import DataInterface, SequentialDataset

torch.manual_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bo_eval_GRU(parameters):
    from DeepMice.models.GRU import GRU, train, test

    net = GRU(example_batch_X=batch,
              x_feat_length=parameters['xfl'],
              gru_hidden_size=parameters['ghs'])
    net = train(model=net,
                train_loader=train_loader,
                valid_loader=val_loader,
                device='cpu',
                learn_rate=parameters['lr'],
                n_epochs=50,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam)

    accuracy = test(net, test_loader=test_loader, device='cpu')

    return accuracy



# def tr_ev(model_func, parameters):
#     net = model_func()
#
# def train_evaluate(parameterization):
#     net = train(net=model, train_loader=train_loader, parameters=parameterization, dtype=dtype, device=device)
#     return evaluate(
#         net=net,
#         data_loader=valid_loader,
#         dtype=dtype,
#         device=device,
#     )
def get_bo_params():
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "xfl", "type": "range", 'value_type': "int", "bounds": [10, 100]},
            {"name": "ghs", "type": "range", 'value_type': "int", "bounds": [3, 20]},
            {"name": "lr", "type": "range", "bounds": [1e-4, 0.1], "log_scale": True},
        ],
        evaluation_function=bo_eval_GRU,
        objective_name='accuracy',
    )

    return best_parameters, values, experiment, model

# best_parameters
# means, covariances = values
# means, covariances
#
# render(plot_contour(model=model, param_x='lr', param_y='momentum', metric_name='accuracy'))

if __name__ == '__main__':
    # Paths
    # ex_path = Path('/Users/mc/PycharmProjects/DeepMice/DeepMice/data/020_excSession_v1_ophys_858863712.nc')    # example file
    ex_path = Path('/mnt/sda5/python_projects/public/DeepMice/018_excSession_v1_ophys_856295914.nc')

    # Load data
    data_interface = DataInterface(path=ex_path)
    dataset_train = SequentialDataset(data_interface,
                                      part='train',  # or 'test', 'val'
                                      len_sequence=15
                                      )

    train_loader = DataLoader(dataset_train,
                              batch_size=5,
                              shuffle=True,
                              worker_init_fn=3453)

    dataset_val = SequentialDataset(data_interface,
                                    part='val',  # or 'test', 'val'
                                    len_sequence=15, )

    val_loader = DataLoader(dataset_val,
                            batch_size=5,
                            shuffle=False,
                            worker_init_fn=3453)

    dataset_test = SequentialDataset(data_interface,
                                     part='test',  # or 'test', 'val'
                                     len_sequence=15, )

    test_loader = DataLoader(dataset_test,
                             batch_size=5,
                             shuffle=False,
                             worker_init_fn=3453)

    X_b, y_b, init_b = next(iter(train_loader))
    batch = X_b

    best_parameters, values, experiment, model = get_bo_params()