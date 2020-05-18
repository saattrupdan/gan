import torch
from torch import nn
import torch.nn.functional as F

from typing import Union
from pathlib import Path

def get_path(path: Union[str, Path] = '.'):
    cwd = Path.cwd()
    if cwd.name == 'src':
        return cwd.parent / path
    else:
        return Path(path)

class TimeSeriesStationariser(nn.Module):
    ''' 
    Makes a time series stationary by standardising every individual feature
    on every individual time step. This requires all time series to be of the
    same length.

    Args:
        X (array-like):
            The feature matrix, of dimension (batch_size, *, *)
    '''

    def __init__(self, X):
        super().__init__()
        X = torch.FloatTensor(X)
        self.means = torch.mean(X, dim = 0, keepdim = True)
        self.stds = torch.std(X, dim = 0, keepdim = True)

    def forward(self, x):
        ''' Makes time series stationary. 

        Args:
            x (PyTorch tensor):
                The input time series, of dimensions
                (batch_size, *, *)

        Returns:
            PyTorch tensor: The standardised tensor, of the same dimensions
        '''
        return (x - self.means) / self.stds

class Accuracy(nn.Module):
    def forward(self, y_pred, y_true):
        return accuracy(y_pred, y_true)

class Mish(nn.Module):
    def forward(self, x):
        return mish(x)

@torch.jit.script
def accuracy(y_pred, y_true):
    preds = torch.argmax(y_pred, dim = 1)
    return torch.mean((preds == y_true).float())

@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))
