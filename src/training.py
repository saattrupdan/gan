import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Union
import itertools as it
from models import RGAN
from tqdm.auto import tqdm

def train_gan(gan: RGAN, dataloader: DataLoader, lr: float = 3e-4,
    lr_reduction_factor: float = 0.99, l2_reg: float = 1e-1, 
    head_start: int = 1, optimiser: type = optim.AdamW, 
    lr_reduction_step_size: int = 100,
    epochs: Union[int, None] = None) -> RGAN:

    # Set up optimiser and scheduler for the critic
    c_optim = optimiser(gan.crt.parameters(), lr = lr)
    c_sched = optim.lr_scheduler.StepLR(
        c_optim, 
        step_size = lr_reduction_step_size, 
        gamma = lr_reduction_factor
    )

    # Set up optimiser and scheduler for the generator
    g_optim = optimiser(gan.gen.parameters(), lr = lr)
    g_sched = optim.lr_scheduler.StepLR(
        g_optim, 
        step_size = lr_reduction_step_size, 
        gamma = lr_reduction_factor
    )

    epoch_it = it.count() if epochs is None else range(epochs)
    with tqdm(epoch_it) as pbar:
        for epoch in pbar:
            avg_c_loss, avg_g_loss = 0, 0

            for reals in dataloader:
    
                # Get the sequence length of the batch
                seq_len = reals[0].shape[1]

                # Reshape from [batch_size, seq_len] to 
                # [seq_len, batch_size, 1]
                reals = reals[0].transpose(0, 1).unsqueeze(2)


                ##########################
                ###  Train the critic  ###
                ##########################

                # Compute Wasserstein loss
                c_optim.zero_grad()
                noise = torch.ones_like(reals)
                noise *= torch.rand(seq_len).unsqueeze(1).unsqueeze(2)
                c_loss = torch.mean(gan(noise) - gan.crt(reals))
                avg_c_loss += float(c_loss) / len(dataloader)

                # Add an l2 regularisation term
                norms = torch.FloatTensor([torch.norm(p) 
                    for p in gan.crt.parameters()])
                c_loss += l2_reg * (norms ** 2).sum()

                # Backprop gradients
                c_loss.backward()
                c_optim.step()
                c_sched.step()

                # Skip training the generator for the first <head_start>
                # many epochs
                if epoch < head_start: continue


                #############################
                ###  Train the generator  ###
                #############################

                # Compute Wasserstein loss
                g_optim.zero_grad()
                noise = torch.randn_like(reals)
                g_loss = -torch.mean(gan(noise))
                avg_g_loss += float(g_loss) / len(dataloader)

                # Backprop gradients
                g_loss.backward()
                g_optim.step()
                g_sched.step()
            
            # Logging
            if epoch % 5 == 0:
                with torch.no_grad():
                    inputs = torch.FloatTensor([[[0]] for _ in range(seq_len)])
                    fake = gan.gen(inputs)
                    fake0 = fake[0][0][0].numpy()
                    fake1 = fake[1][0][0].numpy()
                pbar.set_description(\
                    f'Epoch {epoch:3} - '\
                    f'crt_loss {avg_c_loss:.3f} - '\
                    f'gen_loss {avg_g_loss:.3f} - '\
                    f'fake_pair ({fake0:.3f}, {fake1:.3f}) - '\
                    f'distance: {abs(fake1 - fake0):.3f}'
                )

if __name__ == '__main__':
    import numpy as np
    gan = RGAN()
    rnd = np.random.uniform(-1, 2, size = 1000)
    data = [[n, n+1] for n in rnd]
    dataset = TensorDataset(torch.FloatTensor(data))
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
    train_gan(gan, dataloader)
