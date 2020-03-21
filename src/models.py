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

class RDiscriminator(nn.Module):
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
        return torch.sigmoid(x)

class RGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = RGenerator()
        self.dis = RDiscriminator()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        fake = self.gen(x)
        return self.dis(fake)

    def freeze_discriminator(self):
        for param in self.dis.parameters():
            param.requires_grad = False

    def unfreeze_discriminator(self):
        for param in self.dis.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    gan = RGAN()
    print(gan(5))
