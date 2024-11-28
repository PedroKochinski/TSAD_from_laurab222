import os
import glob
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

from src.utils import *
from src.constants import *


file_prefixes = {
	'SMD': 'machine-1-1_',
	'SMAP': 'P-1_',
	'SMAP_new': 'P-1_',
	'MSL': 'C-1_',
	'MSL_new': 'C-1_',
	'UCR': '136_',
	'NAB': 'ec2_request_latency_system_failure_',
}


def convert_to_windows(data, model): # old version
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if model.name in ['TranAD', 'Attention', 'iTransformer'] else w.view(-1))
	return torch.stack(windows)


def convert_to_windows_new(data, model, window_size=10, step_size=1, ts_lengths=[]):
	"""
	Converts the input time series data into overlapping (except if window_size == step_size) windows.
	Parameters:
	data (torch.Tensor): The input time series data.
	model (object): The model object which contains the model name.
	window_size (int, optional): The size of each window. Default is 10.
	step_size (int, optional): The step size between consecutive windows. Default is 1.
	ts_lengths (list, optional): A list containing the lengths of individual time series. Default is an empty list.
	Returns:
	tuple: A tuple containing:
		- windows (torch.Tensor): The tensor containing the windows of the time series data.
		- ideal_lengths (list): A list containing the ideal lengths of the padded time series.
	"""

	ideal_lengths = []
	if model.name in ['iTransformer'] and step_size > 1: 
		windows = torch.tensor([])
		if ts_lengths == [] or ts_lengths[0] == []:    # check lengths of individual time series, otherwise assume data is one time series
			ts_lengths = [len(data)]
		start = 0
		for l in ts_lengths:
			nb_windows = math.ceil((l - window_size) / step_size) + 1  # get number of complete windows with window size window_size and step size step_size + 1 incomplete
			if nb_windows <= 0: nb_windows = 1                    	   # if not enough data for one window, create one window with padding
			ideal_len = (nb_windows - 1) * step_size + window_size     # length of padded time series
			ideal_lengths.append(ideal_len)
			ts = data[start:start+l]  # separate the individual time series before slicing it into windows to avoid overlap
			# ts = torch.nn.functional.pad(ts, (0, 0, 0, ideal_len - l), 'constant', 0) # zero pad to have a multiple of window_size
			ts = torch.cat((ts, ts[-1].repeat(ideal_len - l, 1)), axis=0)  # pad with last element to have a multiple of window_size
			new_window = torch.stack([ts[i*step_size:i*step_size+window_size] for i in range(nb_windows)])
			windows = torch.cat((windows, new_window), axis=0)
			start += l
	else:  # alternative version where first few windows repeat the first element, then always have 1 new element per window
		windows = convert_to_windows(data, model)

	return windows, ideal_lengths


def load_dataset(dataset, feats=-1, less=False, enc=False):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	ts_lengths = []
	enc_feats = 0

	for file in ['train', 'test', 'labels']:
		if 'IEEECIS' in dataset or 'ATLAS' in dataset:
			paths = glob.glob(os.path.join(folder, f'*{file}*.npy'))
			paths = sorted(paths)  # sort paths to ensure correct order, otherwise labels & test files are mismatched
			if enc:
				enc_paths = glob.glob(os.path.join(folder, f'*timestamp_{file[:2]}_*.npy'))
				enc_paths = sorted(enc_paths)
			if less and file == 'train':
				if dataset in ['ATLAS_TS']:
					paths = [paths[0]]
					if enc:
						enc_paths = [enc_paths[0]]
				elif 'IEEECIS' in dataset:
					paths = paths[:50]
					if enc:
						enc_paths = enc_paths[:50]
			loader.append(np.concatenate([np.load(p) for p in paths]))
			if enc and file != 'labels':
				enc_loader = np.concatenate([np.load(p) for p in enc_paths])
				loader[-1] = np.concatenate((enc_loader, loader[-1]), axis=1)
				enc_feats = enc_loader.shape[1]
			ts_lengths.append([np.load(p).shape[0] for p in paths])
		elif dataset in file_prefixes:
			prefix = file_prefixes[dataset]
			loader.append(np.load(os.path.join(folder, f'{prefix}{file}.npy')))
			ts_lengths.append([loader[-1].shape[0]])
		else:
			loader.append(np.load(os.path.join(folder, f'{file}.npy')))
			ts_lengths.append([loader[-1].shape[0]])

	if dataset in ['SMD', 'IEEECIS'] and less:
		loader[0] = cut_array(0.3, loader[0])
		loader[1] = cut_array(0.1, loader[1])
		loader[2] = cut_array(0.1, loader[2])
		ts_lengths = [[loader[i].shape[0]] for i in range(len(loader))]  # update time series lengths 
	elif less and 'IEEECIS' not in dataset and 'ATLAS' not in dataset:
		loader[0] = cut_array(0.5, loader[0])
		loader[1] = cut_array(0.3, loader[1])
		loader[2] = cut_array(0.3, loader[2])
		ts_lengths = [[loader[i].shape[0]] for i in range(len(loader))]  # update time series lengths 

	if feats > 0:  # reduce number of features
		print(f'data set has {loader[0].shape[1]} features, only using {feats}')
		for i in range(2):
			max_feats = feats + enc_feats
			loader[i] = loader[i][:,:max_feats]
	
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	
	if labels.shape[1] == 1: # if labels are 1D, repeat them for each feature to have 2D labels
		labels = np.repeat(labels, loader[0].shape[1], axis=1)
		
	print('training set shape:', train_loader.dataset.shape)
	print('test set shape:', test_loader.dataset.shape)
	print('labels shape:', labels.shape)
	print('ts_lengths 0:', np.sum(ts_lengths[0]))
	print('ts_lengths 1:', np.sum(ts_lengths[1]))
	return train_loader, test_loader, labels, ts_lengths, enc_feats


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

# 		df_stamp = df_raw[['date']][border1:border2]
# 		df_stamp['date'] = pd.to_datetime(df_stamp.date)
# 		if self.timeenc == 0:
# 			df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
# 			df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
# 			df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
# 			df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
# 			data_stamp = df_stamp.drop(['date'], 1).values
# 		elif self.timeenc == 1:
# 			data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
# 			data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)