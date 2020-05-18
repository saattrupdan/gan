import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

from typing import Optional
import itertools as it
from tqdm.auto import tqdm

from data_prep import load_hdf
from utils import get_path, TimeSeriesStationariser, Accuracy
from tsc_model import TSCResNet

def train_model(
    model: nn.Module, 
    epochs: Optional[int] = None, 
    batch_size: int = 32,
    val_size: Optional[float] = 0.1, 
    learning_rate: float = 3e-3,
    momentum: float = 0.9, 
    second_momentum: float = 0.999,
    weight_decay: float = 0.01, 
    lr_reduction_factor: float = 0.3,
    lr_reduction_patience: int = 10, 
    patience: int = 30,
    datadir: str = '.data', 
    filename: str = 'ElectricDevices',
    min_improvement: float = 1e-3):
    ''' Train the model.

    Args:
        model (PyTorch module):
            The model that is to be trained
        epochs (int or None):
            The number of epochs to train for. If None then will train
            indefinitely. Defaults to None.
        batch_size (int):
            The number of time series to use at a time. Defaults to 32.
        val_size (float or None):
            The proportion of the data set used for validation. If set to 0.0
            or None then no validation set will be used. Defaults to 0.1.
        learning_rate (float):
            The learning rate of the AdamW optimiser. Defaults to 3e-4.
        momentum (float):
            The first momentum of the AdamW optimiser. Defaults to 0.9.
        second_momentum (float):
            The second momentum of the AdamW optimiser. Defaults to 0.999.
        weight_decay (float):
            The amount of weight decay used in the AdamW optimiser. Defaults
            to 0.01.
        lr_reduction_factor (float):
            The factor that will be multiplied by the learning rate when
            validation hasn't improved for ``lr_reduction_patience`` many steps.
            Defaults to 0.1.
        lr_reduction_patience (int):
            The amount of epochs to allow no improvement in the validation
            score before reducing the learning rate by ``lr_reduction_factor``.
            Defaults to 10.
        patience (int or None):
            The number of epochs to allow no improvement in the validation
            score before stopping training. If None then no early stopping
            is used. Defaults to 100.
        datadir (str):
            The name of the data directory, from where it fetches the training
            data and where the model will be stored. Defaults to '.data'.
        filename (str):
            The name of the file to which the model will be saved. Defaults
            to 'ElectricDevices'
        min_improvement (float):
            Threshold for measuring a new best loss. A loss counts as being 
            the best if it's below previous_best - min_improvement. Defaults 
            to 1e-3.

    Returns:
        Tuple of two lists of length ``epochs``, the first one containing 
        the training scores and the second one containing the validation
        scores.
    '''

    # Warn the user if a model already exists
    user_response = 'y'
    if list(get_path(datadir).glob(f'{filename}*.zip')) != []:

        message = f'There is already a model stored called "{filename}" '\
                  f'in "{str(get_path(datadir))}", which will be overwritten.'\
                  f'\nDo you want to continue? (y/n)\n>>> '

        user_response = input(message)
        while user_response not in ['y', 'n']:
            user_response = input('Invalid input, please try again.\n>>> ')
    
    if user_response == 'n':
        return [], []

    elif user_response == 'y':

        #=====================
        # Setting up the data
        #=====================

        # Fetch the data and prepare it
        hdf = load_hdf(filename = filename, datadir = datadir)

        # Convert the NumPy arrays into PyTorch tensors
        X = torch.FloatTensor(hdf['X_train']).unsqueeze(1)
        y = torch.LongTensor(hdf['y_train']) - 1

        # Convert the NumPy arrays X and y into a PyTorch data set
        dataset = TensorDataset(X, y)

        # Split the data set into a training- and testing set
        if val_size is not None and val_size > 0:
            val_size = int(val_size * len(dataset))
            train_size = len(dataset) - val_size
            train, val = random_split(dataset, [train_size, val_size])

            # Pack the data sets into data loaders, enabling iteration
            train_dl = DataLoader(train, batch_size = batch_size,
                                  shuffle = True)
            val_dl = DataLoader(val, batch_size = batch_size, 
                                shuffle = True)

        else:
            train_dl = DataLoader(dataset, batch_size = batch_size, 
                                  shuffle = True)

        #======================================
        # Setting up objects used for training
        #======================================

        # Build stationariser and attach it to X
        stat = TimeSeriesStationariser(X)
        model = nn.Sequential(stat, model)

        # Set the optimiser to be AdamW, which is Adam with weight decay
        optimiser = optim.AdamW(
            model.parameters(), 
            lr = learning_rate, 
            weight_decay = weight_decay,
            betas = (momentum, second_momentum)
        )

        # Set the learning rate scheduler to reduce the learning rate when
        # the validation performance isn't improving
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode = 'max',
            factor = lr_reduction_factor,
            patience = lr_reduction_patience,
            threshold = min_improvement,
            threshold_mode = 'abs'
        )

        # Set the loss function
        criterion = nn.CrossEntropyLoss()

        # Set the metric
        metric = Accuracy()

        # Initialise lists that will store the training scores and
        # the validation scores, for later inspection
        train_scores = []
        val_scores = []

        # Initialise the number of "bad epochs", being epochs with no progress
        bad_epochs = 0

        # Initialise the best validation score, which starts by being the 
        # worst possible
        best = 0.

        # Output model data
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f'Number of trainable model parameters: {params:,}')

        # Define the epoch iterator and set up progress bar if `epochs` is set
        if epochs is not None:
            epoch_mode = True
            epochs = tqdm(range(epochs), desc = 'Training model')
        else:
            epoch_mode = False
            epochs = it.count()

        # Main training loop
        for epoch in epochs:

            #==========
            # Training
            #==========

            # Enable training mode for the model, which enables dropout and
            # ensures that batch normalisation layers compute batch-wise means
            # and standard deviations
            model.train()

            # Reset the scores
            train_score = 0.
            val_score = 0.
       
            # Set progress bar description
            if not epoch_mode: 
                pbar = tqdm(total = train_size, desc = f'Epoch {epoch}')

            # Iterate through the training data and teach the model what 
            # not to do
            for idx, (x_train, y_train) in enumerate(train_dl):

                # Reset all the gradients stored in the optimiser
                optimiser.zero_grad()

                # Get the model's predictions for the batch
                y_hat = model.forward(x_train)

                # Compute the loss. This indicates how badly the model 
                # performed in its predictions
                loss = criterion(y_hat, y_train)

                # PyTorch automatically calculates the gradients on the fly, so
                # here we backpropagate the gradients for the computed loss
                loss.backward()

                # Using these gradients, the optimiser changes the weights in
                # the model ever so slightly, towards better performance.
                # It's important that we called `zero_grad` before, to make 
                # sure that this change is only affected by the gradients from 
                # the current batch
                optimiser.step()

                # Compute training score
                train_score += metric(y_hat, y_train)

                # Update progress bar
                if not epoch_mode: pbar.update(x_train.shape[0])

                # Set progress bar description
                pbar.set_description(
                    f'Epoch {epoch} - train acc {train_score / (idx + 1):.4f}'
                )

            # Store the mean training loss for later inspection
            train_score /= len(train_dl)
            train_scores.append(train_score)


            #============
            # Evaluation
            #============

            if val_size is not None and val_size > 0:

                # Set progress bar description
                if not epoch_mode: 
                    pbar.set_description(f'Epoch {epoch} - Evaluating...')

                # Enable validation mode for the model, which disables dropout
                # and makes batch normalisation layers use their stored values
                # for mean and standard deviation
                model.eval()

                # Disable PyTorch's automatic calculation of gradients, as we 
                # don't need that for evaluation
                with torch.no_grad():

                    # Iterate through the validation data to evaluate the 
                    # model's performance on data that it hasn't seen before. 
                    for x_val, y_val in val_dl:
                        
                        # Get the model's predictions for the batch
                        y_hat = model.forward(x_val)

                        # Compute the metric
                        val_score += metric(y_hat, y_val) / len(val_dl)

                # Update the learning rate scheduler, which will reduce the
                # learning rate if no progress has been made for a while
                scheduler.step(val_score)


            #=============
            # Bookkeeping
            #=============

            # If score is best so far
            if val_scores == [] or val_score > best:

                # Set new best
                best = val_score + min_improvement

                # Remove previously saved models
                for fname in get_path(datadir).glob(f'{filename}*.zip'):
                    fname.unlink()

                # Reset number of epochs with no progress
                bad_epochs = 0

                # Save scripted version of the model
                path = get_path(datadir) / f'{filename}-{val_score:.4f}.zip'
                scripted_model = torch.jit.script(model)
                scripted_model.save(str(path))
            else:
                bad_epochs += 1

            # Update the progress bars
            pbar_desc = f' - train acc {train_score:.4f}'\
                        f' - val acc {val_score:.4f}'\
                        f' - bad epochs {bad_epochs}'
            if epoch_mode:
                epochs.set_description('Training model' + pbar_desc)
            else:
                pbar.set_description(f'Epoch {epoch}' + pbar_desc)
                pbar.close()

            # Store the score for later inspection
            val_scores.append(val_score)

            # Early stopping after `patience` epochs with no progress
            if bad_epochs > patience: break

        return train_scores, val_scores

def evaluate_tsc(filename: str = 'ElectricDevices', datadir: str = '.data'):
    with torch.no_grad():
        # Load model
        path = next(get_path(datadir).glob(f'{filename}*.zip'))
        model = torch.jit.load(str(path))
        model.eval()

        # Fetch the data and prepare it
        hdf = load_hdf(filename = filename, datadir = datadir)

        # Convert the NumPy arrays into PyTorch tensors
        X = torch.FloatTensor(hdf['X_test']).unsqueeze(1)
        y = torch.LongTensor(hdf['y_test']) - 1

        # Convert the NumPy arrays X and y into a PyTorch data set
        dataset = TensorDataset(X, y)

        # Pack the data sets into data loaders, enabling iteration
        dl = DataLoader(dataset, batch_size = 32)

        # Set up progress bar
        with tqdm(total = len(dl) * dl.batch_size, desc = 'Evaluating') as pbar:

            # Calculate mean accuracy
            acc_fn = Accuracy()
            acc = 0.
            for x, y in dl:
                yhat = model(x)
                acc += acc_fn(yhat, y)
                pbar.update(dl.batch_size)
            acc /= len(dl)

        return acc

if __name__ == '__main__':
    model = TSCResNet()
    train_model(model)
    #print(evaluate_tsc())
