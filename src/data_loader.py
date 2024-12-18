import os
import math
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, dataset, window_size, step_size, modelname, flag='train', feats=-1, less=False, enc=False, k=-1):
        """
        Initializes the data loader with the specified parameters.
        Args:
            dataset (str): The name of the dataset to load.
            window_size (int): The size of the window for creating data segments.
            step_size (int): The step size for moving the window.
            modelname (str): The name of the model to be used.
            flag (str, optional): The type of data to load (e.g., 'train', 'test'). Defaults to 'train'.
            feats (int, optional): The number of features in the dataset. Defaults to None (then automatically == nb of input features).
            less (bool, optional): A flag indicating whether to load a smaller subset (10k timestamps) of the data. Defaults to False.
            enc (bool, optional): A flag indicating whether to use encoding of timestamp. Defaults to False.
            k (int, optional): Whether to do a 5-fold cross validation, k indicates which fold to use for validation. Defaults to -1.
        Methods:
            __load_data__(type): Loads the data based on the specified type.
            __make_windows__(data): Creates windows of data segments based on the window size.
        """

        self.data_name = dataset
        self.modelname = modelname

        self.window_size = window_size
        self.step_size = step_size
        self.feats = feats
        self.enc = enc
        
        self.flag = flag
        self.less = less
        assert k < 5
        self.k = k

        self.__load_data__(type=flag)
        self.feats = self.data.shape[1] if feats == 0 else feats

        if self.window_size > 0:
            self.complete_data = self.data
            self.__make_windows__(self.data)

    def __load_data__(self, type='train'):
        folder = os.path.join('processed', self.data_name)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')

        file = 'train' if type == 'valid' else type

        paths = glob.glob(os.path.join(folder, f'*{file}*.npy'))
        paths = sorted(paths)  # sort paths to ensure correct order, otherwise labels & test files are mismatched
        if self.k > 0 and len(paths) > 1:
            n = len(paths) // 5
            if type == 'train':
                paths = paths[:self.k*n] + paths[(self.k+1)*n:]
            elif type == 'valid':
                paths = paths[self.k*n:(self.k+1)*n] 
        data = np.concatenate([np.load(p) for p in paths])
        ts_lengths = [np.load(p).shape[0] for p in paths]

        if type == 'test':
            l_paths = glob.glob(os.path.join(folder, f'*labels*.npy'))
            labels = np.concatenate([np.load(p) for p in l_paths])
            print(labels.shape, labels[0].shape)

        if self.feats > 0:
            data = data[:, :self.feats]

        if self.less:
            data = data[:10000]
            ts_lengths = [data.shape[0]]
            if type == 'test':  
                labels = labels[:10000]

        # 5-fold cross validation
        if self.k > 0 and len(paths) == 1:
            n = data.shape[0] // 5
            if type == 'train':
                data = np.concatenate([data[:self.k*n], data[(self.k+1)*n:]])
                ts_lengths = [data.shape[0]]
            elif type == 'valid':
                data = data[self.k*n:(self.k+1)*n]
                ts_lengths = [data.shape[0]]

        self.data = data
        self.ts_lengths = ts_lengths
        if type == 'test':
            self.labels = labels

    def __make_windows__(self, data):
        """
        Converts the input time series data into overlapping (except if window_size == step_size) windows.
        Parameters:
        data (np.array): The input time series data.
        Creates:
        tuple: A tuple containing:
            - windows (np.array): The tensor containing the windows of the time series data.
            - ideal_lengths (list): A list containing the lengths of the padded time series.
        """

        ideal_lengths = []
        if 'iTransformer' in self.modelname or self.modelname in ['LSTM_AE']: 
            windows = np.empty((0, self.window_size, data.shape[1]))
            start = 0
            for l in self.ts_lengths:
                # get number of complete windows with window size window_size and step size step_size + 1 incomplete
                nb_windows = math.ceil((l - self.window_size) / self.step_size) + 1  
                if nb_windows <= 0: nb_windows = 1            
                # if not enough data for one window, create one window with padding, ideal_len: length of padded time series        	   
                ideal_len = (nb_windows - 1) * self.step_size + self.window_size    
                ideal_lengths.append(ideal_len)
                # separate the individual time series before slicing it into windows to avoid overlap
                ts = data[start:start+l]  
                if ideal_len > l: # pad with last element to have a multiple of window_size
                    ts = np.concatenate((ts, ts[-1].repeat(ideal_len - l, 1)), axis=0)  
                new_window = np.stack([ts[i*self.step_size:i*self.step_size+self.window_size] for i in range(nb_windows)])
                windows = np.concatenate((windows, new_window), axis=0)
                start += l
            self.data = windows
            self.ideal_lengths = ideal_lengths
        else:  # alternative version where first few windows repeat the first element, then always have 1 new element per window
            windows = []; 
            for i, g in enumerate(data): 
                if i >= self.window_size: w = data[i-self.window_size:i]
                else: w = np.concatenate([data[0].repeat(self.window_size-i, 1), data[0:i]])
                windows.append(w if self.modelname in ['TranAD', 'Attention', 'iTransformer', 'LSTM_AE'] else w.view(-1))
            windows = np.stack(windows)
            self.data = windows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    
    def get_ideal_lengths(self):
        return self.ideal_lengths
    
    def get_labels(self):
        assert self.flag == 'test'
        # if labels are 1D, repeat them for each feature to have 2D labels
        if self.feats != self.labels.shape[1]:
            self.labels = np.repeat(self.labels, self.feats, axis=1)
        return self.labels
    
    def get_complete_data(self):
        # for plots or if we want to use unsliced data
        return self.complete_data


if __name__ == '__main__':
    dataset = 'SMD'
    # Create dataset
    train = MyDataset(dataset, window_size=10, step_size=1, modelname='iTransformer', flag='train', feats=30, less=False, enc=False, k=1)
    valid = MyDataset(dataset, window_size=10, step_size=1, modelname='iTransformer', flag='valid', feats=30, less=False, enc=False, k=1)
    test = MyDataset(dataset, window_size=10, step_size=1, modelname='iTransformer', flag='test', feats=30, less=False, enc=False, k=-1)
    print(train.__len__(), train.data.shape, train.complete_data.shape)
    print(valid.__len__(), valid.data.shape)
    print(test.__len__(), test.data.shape)
    labels = test.get_labels()
    print(labels.shape)

    # Create data loader
    data_loader_train = DataLoader(train, batch_size=24, shuffle=True)
    data_loader_test = DataLoader(test, batch_size=24, shuffle=True)

    # # Iterate through the data loader
    for batch in data_loader_train:
        print(batch.shape)
    for batch in data_loader_test:
        print(batch.shape)