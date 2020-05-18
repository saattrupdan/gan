import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional

class RGenerator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.rnn = nn.GRU(1, dim, num_layers = 2)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x):

        # (seq_len, batch_size, 1) -> (seq_len, batch_size, dim)
        x, _ = self.rnn(x)

        # (seq_len, batch_size, dim) -> (seq_len, batch_size, 1)
        return self.proj(x)

class RCritic(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.rnn = nn.GRU(1, dim)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x):

        # (seq_len, batch_size, 1) -> (batch_size, dim)
        _, x = self.rnn(x)

        # (batch_size, dim) -> (batch_size, 1)
        return self.proj(x)

class RGAN(nn.Module):
    ''' An implementation of a Recurrent Generative Adversarial Network. '''
    def __init__(self, gen_dim: int = 32, crt_dim: int = 32, 
                 random_seed: Optional[int] = None):
        torch.manual_seed(random_seed)
        super().__init__()
        self.gen = RGenerator(dim = gen_dim)
        self.crt = RCritic(dim = crt_dim)

    def forward(self, x):
        fake = self.gen(x)
        return self.crt(fake)

if __name__ == '__main__':
    gan = RGAN()
    print(gan(5))
