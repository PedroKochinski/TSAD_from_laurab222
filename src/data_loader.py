import os
import glob
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

from src.utils import *
from src.constants import *


def convert_to_windows(data, model): # old version
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if model.name in ['TranAD', 'Attention', 'iTransformer'] else w.view(-1))
	return torch.stack(windows)


def convert_to_windows_train(data, model, step_size=1, ts_lengths=[]):
	""" convert data to windows
	if step_size > 1, create overlapping windows shifted by step_size (if step_size = w_size, no overlap)
	or if step_size = 1, only one new element per window with all the rest of the window overlapping """
	
	w_size = model.n_window
	
	if model.name in ['Attention', 'iTransformer']: 
		windows = torch.tensor([])
		if ts_lengths == [] or ts_lengths[0] == []:    # check lengths of individual time series, otherwise assume data is one time series
			ts_lengths = [len(data)]
		else:
			ts_lengths = ts_lengths[0]  # because 1st element contains lengths of train TS, 2nd of test TS
		start = 0
		for l in ts_lengths:
			nb_windows = math.ceil((l - w_size) / step_size)  # get number of complete windows with window size w_size and step size step_size 
			if nb_windows <= 0: nb_windows = 1                 # if not enough data for one window, create one window with zero padding
			ideal_len = nb_windows * w_size
			ts = data[start:start+l]  # separate the individual time series before slicing it into windows to avoid overlap
			ts = torch.nn.functional.pad(ts, (0, 0, 0, ideal_len - l), 'constant', 0) # zero pad to have a multiple of w_size
			new_window = torch.stack([ts[i*step_size:i*step_size+w_size] for i in range(nb_windows)])
			windows = torch.cat((windows, new_window), axis=0)
			start += l
	else:  # alternative version where first few windows repeat the first element, then always have 1 new element per window
		windows = convert_to_windows(data, model)

	return windows


def convert_to_windows_test(data, model, labels=None, w_size=10):
		
	if model.name in ['Attention', 'iTransformer']: 
		windows = torch.tensor([])
		l = len(data)
		nb_windows = math.ceil(l / w_size)  # get number of complete windows with window size w_size
		if nb_windows <= 0: nb_windows = 1                 # if not enough data for one window, create one window with zero padding
		ideal_len = nb_windows * w_size
		if labels is not None:
			labels = np.pad(labels,((0, ideal_len - l), (0, 0)), 'constant', constant_values=0) # zero pad such that labels have same length as test
		data = torch.nn.functional.pad(data, (0, 0, 0, ideal_len - l), 'constant', 0) # zero pad to have a multiple of w_size
		new_window = torch.stack([data[i*w_size:(i+1)*w_size] for i in range(nb_windows)])
		windows = torch.cat((windows, new_window), axis=0)
	
	else:  # alternative version where first few windows repeat the first element, then always have 1 new element per window
		windows = convert_to_windows(data, model)

	return windows, labels


def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	ts_lengths = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		elif dataset == 'SMAP': file = 'P-1_' + file
		elif dataset == 'SMAP_new': file = 'P-1_' + file
		elif dataset == 'MSL': file = 'C-1_' + file
		elif dataset == 'MSL_new': file = 'C-1_' + file
		elif dataset == 'UCR': file = '136_' + file
		elif dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		elif dataset == 'IEEECIS': file = file + '_1'		# first naive version of this
		elif dataset == 'IEEECIS_new': 					    # time series of users
			paths = glob.glob(os.path.join(folder, f'{file}_*.npy'))
			paths = sorted(paths)               # sort paths to ensure correct order, otherwise labels & test files are mismatched
			loader.append(np.concatenate([np.load(p) for p in paths]))
			ts_lengths.append([np.load(p).shape[0] for p in paths])
			# if file=='train': file = file + '_1137.0_299.0_-480.0'
			# else: file = file + '_15885.0_nan_176.0'
		elif dataset == 'ATLAS_TS': 
			if file=='train': file = 'lb_0_' + file
			else: file = 'lb_1_' + file
		
		if dataset not in ['IEEECIS_new']:
			loader.append(np.load(os.path.join(folder, f'{file}.npy')))

	if dataset in ['SMD', 'IEEECIS'] and args.less:
		loader[0] = cut_array(0.02, loader[0])
		loader[1] = cut_array(0.02, loader[1])
		loader[2] = cut_array(0.02, loader[2])
	elif args.less: 
		loader[0] = cut_array(0.5, loader[0])
		loader[1] = cut_array(0.3, loader[1])
		loader[2] = cut_array(0.3, loader[2])
	
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	
	if labels.shape[1] == 1: # if labels are 1D, repeat them for each feature to have 2D labels
		labels = np.repeat(labels, loader[0].shape[1], axis=1)
		
	print('training set shape:', train_loader.dataset.shape)
	print('test set shape:', test_loader.dataset.shape)
	print('labels shape:', labels.shape)
	return train_loader, test_loader, labels, ts_lengths