import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from models import RGAN

def train_gan(gan: RGAN, dataloader: DataLoader, epochs: int = 1000,
    lr: float = 4e-3) -> RGAN:

    criterion = nn.BCELoss()
    optim = AdamW(gan.parameters(), lr = lr)

    seq_len = next(iter(dataloader))[0].shape[1]
    batch_size = dataloader.batch_size

    for epoch in range(epochs):
        avg_dis_loss, avg_gen_loss = 0, 0

        for reals in dataloader:
            # Reshape from [batch_size, seq_len] to [seq_len, batch_size, 1]
            reals = reals[0].transpose(0, 1).unsqueeze(2)

            # Train the discriminator
            optim.zero_grad()
            noise = torch.randn_like(reals)
            fakes = gan.gen(noise)
            gan.unfreeze_discriminator()

            preds_fake = gan.dis(fakes)
            y_fake = torch.zeros_like(preds_fake)
            dis_loss_fake = criterion(preds_fake, y_fake)

            preds_real = gan.dis(reals)
            y_real = torch.ones_like(preds_real)
            dis_loss_real = criterion(preds_real, y_real)

            dis_loss = (dis_loss_fake + dis_loss_real) / 2
            dis_loss.backward()
            preds_real = gan.dis(reals)
            optim.step()

            # Train the generator
            optim.zero_grad()
            noise = torch.randn_like(reals)
            gan.freeze_discriminator()
            preds = gan(noise)
            y = torch.ones_like(preds)
            gen_loss = criterion(preds, y)
            gen_loss.backward()
            optim.step()

            # Average losses over the epoch
            avg_dis_loss += float(dis_loss) / len(dataloader)
            avg_gen_loss += float(gen_loss) / len(dataloader)

        if epoch % 100 == 0:
            print(gan.gen(torch.randn(seq_len, 1, 1)))

        # Logging
        print(f'Epoch {epoch:2d} - '\
              f'dis_loss {avg_dis_loss:.4f} - '\
              f'gen_loss {avg_gen_loss:.4f}', end = '\r')

if __name__ == '__main__':
    gan = RGAN()
    data = [list(range(n, n+3)) for n in range(1000)]
    dataset = TensorDataset(torch.FloatTensor(data))
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
    train_gan(gan, dataloader)
