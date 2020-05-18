import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

import itertools as it
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union
import os
import time

from models import RGAN


def train_gan(gan: RGAN, dataloader: DataLoader, lr: float = 3e-4,
    lr_reduction_factor: float = 1., lr_reduction_step: int = 10,
    l2_reg: float = 0., optimiser: type = optim.AdamW, 
    epochs: Union[int, None] = None, model_name: str = 'walmart_gan',
    datadir: str = '.data', random_seed: int = 42,
    tensorboard_run: str = 'loss') -> RGAN:

    # Set random seed
    if random_seed is not None: torch.manual_seed(random_seed)

    # Initialise tensorboard
    writer= SummaryWriter()
    os.system('tensorboard --logdir runs &')
    time.sleep(4)

    # Set up optimiser and scheduler for the critic
    c_optim = optimiser(gan.crt.parameters(), lr = lr)
    c_sched = optim.lr_scheduler.StepLR(
        c_optim, 
        step_size = lr_reduction_step, 
        gamma = lr_reduction_factor
    )

    # Set up optimiser and scheduler for the generator
    g_optim = optimiser(gan.gen.parameters(), lr = lr)
    g_sched = optim.lr_scheduler.StepLR(
        g_optim, 
        step_size = lr_reduction_step, 
        gamma = lr_reduction_factor
    )

    epoch_it = it.count() if epochs is None else range(epochs)
    for epoch in epoch_it:
        avg_c_loss, avg_g_loss = 0, 0

        with tqdm(total = len(dataloader) * dataloader.batch_size,
                  desc = f'Epoch {epoch:3}') as pbar:

            for idx, reals in enumerate(dataloader):
                
                # [batch_size, seq_len] -> [seq_len, batch_size, 1]
                reals = reals[0].transpose(0, 1).unsqueeze(2)

                # Get the batch size and sequence length of the batch
                seq_len, batch_size, _ = reals.shape


                ##########################
                ###  Train the critic  ###
                ##########################

                # Compute Wasserstein loss
                c_optim.zero_grad()
                noise = torch.randn_like(reals)
                c_loss = torch.mean(gan(noise) - gan.crt(reals))

                # Add an l2 regularisation term
                norms = torch.FloatTensor([torch.norm(p) 
                    for p in gan.crt.parameters()])
                c_loss += l2_reg * (norms ** 2).sum()

                # Compute average loss for logging
                avg_c_loss += float(c_loss) / len(dataloader)

                # Backprop gradients
                c_loss.backward()
                c_optim.step()


                #############################
                ###  Train the generator  ###
                #############################

                # Compute Wasserstein loss
                g_optim.zero_grad()
                noise = torch.randn_like(reals)
                g_loss = -torch.mean(gan(noise))

                # Compute average loss for logging
                avg_g_loss += float(g_loss) / len(dataloader)

                # Backprop gradients
                g_loss.backward()
                g_optim.step()

                # Update progress bar
                pbar.update(dataloader.batch_size)

                # Add to tensorboard
                metrics = {'gen': float(g_loss), 'crt': float(c_loss)}
                niter = epoch * len(dataloader) + (idx + 1)
                writer.add_scalars(tensorboard_run, metrics, niter)
          

            # Update learning rate schedulers
            c_sched.step()
            g_sched.step()

            # Logging
            pbar.set_description(\
                f'Epoch {epoch:3} - '\
                f'crt_loss {avg_c_loss:.3f} - '\
                f'gen_loss {avg_g_loss:.3f}'
            )

        # Save model
        path = Path.cwd().parent / datadir / f'{model_name}.zip'
        scripted_gan = torch.jit.script(gan)
        scripted_gan.save(str(path))

    writer.close()
    os.system('killall -9 tensorboard')

if __name__ == '__main__':
    from data_prep import load_hdf
    import matplotlib.pyplot as plt

    hdf = load_hdf()
    X = torch.FloatTensor(hdf['X'])
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

    SEED = 42
    gan = RGAN(gen_dim = 256, crt_dim = 64, random_seed = SEED)
    train_gan(gan, dataloader, random_seed = SEED)#, l2_reg = 0.1, 
              #lr_reduction_factor = 0.9, lr_reduction_step = 10)

    #with torch.no_grad():
    #    gan = torch.jit.load('../.data/walmart_gan.zip')
    #    noise = torch.randn(143, 10, 1)
    #    fakes = gan(noise).squeeze().transpose(0, 1)

    #rnds = torch.randint(0, X.size(0), (10,))
    #reals = X[rnds, :]

    #xs = torch.arange(X.shape[1])
    #plt.figure(figsize = (10, 5))

    #for real in reals:
    #    plt.plot(xs, real)
    #plt.show()

    #for fake in fakes:
    #    plt.plot(xs, fake)
    #    plt.show()
