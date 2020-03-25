import torch
from torch import nn
import torch.nn.functional as F

class RGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x, _ = self.rnn(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class RCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x, _ = self.rnn(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class RGAN(nn.Module):
    ''' An implementation of a Recurrent Generative Adversarial Network. '''
    def __init__(self):
        super().__init__()
        self.gen = RGenerator()
        self.crt = RCritic()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        fake = self.gen(x)
        return self.crt(fake)

if __name__ == '__main__':
    gan = RGAN()
    print(gan(5))
