import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from tqdm.auto import tqdm
import h5py
import io

def build_walmart_hdf(filename: str = 'walmart', datadir: str = '.data'):

    datadir = Path.cwd().parent / datadir
    cols = ['Store', 'Dept', 'Date', 'Weekly_Sales']
    df = pd.read_csv(datadir / f'{filename}.csv')

    store_depts = set(zip(df.Store, df.Dept))
    pbar = tqdm(store_depts, desc = 'Finding maximum sequence length')

    df['Date'] = pd.to_datetime(df.Date)
    df = df.pivot_table(values = 'Weekly_Sales', 
                        index = ['Store', 'Dept', 'Date'])

    max_seq_len = max(len(df.loc[store, dept]) for store, dept in pbar)
    X = -np.ones((len(store_depts), max_seq_len))

    pbar = tqdm(store_depts, desc = 'Creating dataset')
    for idx, (store, dept) in enumerate(pbar):
        series = df.loc[store, dept].squeeze()
        if isinstance(series, float): series = [series]
        X[idx, :len(series)] = series

    with h5py.File(datadir / f'{filename}.h5', 'w') as hdf:
        hdf['X'] = X

    return X

def build_energy_hdf(filename: str = 'ElectricDevices', datadir: str = '.data'):
    ''' 
    These problems were taken from data recorded as part of government 
    sponsored study called Powering the Nation. The intention was to collect 
    behavioural data about how consumers use electricity within the home to 
    help reduce the UK's carbon footprint. The data contains readings from 251 
    households, sampled in two-minute intervals over a month. Each series is 
    length 720 (24 hours of readings taken every 2 minutes).

    Source:
        https://www.timeseriesclassification.com/description.php?Dataset=ElectricDevices
    '''

    datadir = Path.cwd().parent / datadir

    with open(datadir / f'{filename}_TRAIN.txt', 'r') as f:
        train_data = '\n'.join([line[3:].rstrip().replace('  ', ',') 
                                for line in f.readlines()])
    train_csv = io.StringIO(train_data)
    train_df = pd.read_csv(train_csv, header = None)
    train_df = train_df[list(range(1, 96)) + [0]]

    X_train = train_df.values[:, :95]
    y_train = train_df.values[:, 95]

    with open(datadir / f'{filename}_TEST.txt', 'r') as f:
        test_data = '\n'.join([line[3:].rstrip().replace('  ', ',') 
                                for line in f.readlines()])
    test_csv = io.StringIO(test_data)
    test_df = pd.read_csv(test_csv, header = None)
    test_df = test_df[list(range(1, 96)) + [0]]

    X_test = test_df.values[:, :95]
    y_test = test_df.values[:, 95]

    with h5py.File(datadir / f'{filename}.h5', 'w') as hdf:
        hdf['X_train'] = X_train
        hdf['y_train'] = y_train
        hdf['X_test'] = X_test
        hdf['y_test'] = y_test

    return X_train, y_train, X_test, y_test

def load_hdf(filename: str = 'walmart', datadir: str = '.data'):
    datadir = Path.cwd().parent / datadir
    return h5py.File(datadir / f'{filename}.h5', 'r')

if __name__ == '__main__':
    build_energy_hdf()
