import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder  
from src.folderconstants import *


datasets = ['creditcard_normal', 'GECCO_normal', 'IEEECIS', 'MSL', 
			'SMAP', 'SMD', 'SWaT', 'SWaT_1D' 'UCR', 'WADI']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = 0, max_a = 1):  # min_a = None, max_a = None
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df, reduce=False):
	if reduce:
		x = df.values[::10, :]  # downsampling, only keep 1/10 of data
	else:
		x = df.values
	return (x - x.min(0)) / (np.ptp(x, axis=0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset in ['creditcard', 'creditcard_normal']: # from creditcard kaggle challenge
		data = pd.read_csv('data/creditcard/creditcard.csv')
		data.sort_values(by='Time', inplace=True)
		X = data.drop(['Time', 'Class'], axis=1)
		y = data.Class
		time = data.Time
		features = X.columns
		cut = int(len(X)*0.7)
		X_train = X[:cut]
		X_test = X[cut:]
		y_train = y[:cut]
		y_test = y[cut:]
		X_train = np.array(X_train)
		X_test = np.array(X_test)
		y_train = np.array(y_train)
		y_test = np.array(y_test)
		# take out anomalies of train data (for creditcard_normal)
		if dataset == 'creditcard_normal':
			idx = np.where(y_train == 0)
			X_train = X_train[idx]
			y_train = y_train[idx]
		y_train = y_train[:, np.newaxis]
		y_test = y_test[:, np.newaxis]
		# clip data to [1, 99] percentile of train data
		for dim in range(len(features)):
			x_up = np.percentile(X_train[:,dim], 99)
			x_low = np.percentile(X_train[:,dim], 1)
			X_train[:,dim] = np.clip(X_train[:,dim], x_low, x_up)
			X_test[:,dim] = np.clip(X_test[:,dim], x_low, x_up)
		# only scale 'amount' (last) feature 
		scaler = StandardScaler()
		X_train_scaled = X_train
		X_test_scaled = X_test
		X_train_scaled[:, -1:] = scaler.fit_transform(X_train[:, -1:])
		X_test_scaled[:, -1:] = scaler.transform(X_test[:, -1:])
		print(X_train_scaled.shape, X_test_scaled.shape, y_test.shape)
		np.save(f'{output_folder}/{dataset}/train.npy', X_train_scaled)
		np.save(f'{output_folder}/{dataset}/test.npy', X_test_scaled)
		np.save(f'{output_folder}/{dataset}/labels.npy', y_test)
	elif dataset in ['GECCO', 'GECCO_normal']:  # from GECCO IOT challenge 2018
		data = pd.read_csv('data/GECCO/1_gecco2018_water_quality.csv')
		data = data.sort_values(by='Time')
		X = data.drop(['Time', 'Unnamed: 0', 'EVENT'], axis=1)
		y = data.EVENT
		X = np.array(X)
		y = np.array(y+0)
		# replace nan values with previous value (forward fill)
		for i in range(X.shape[1]):
			for j in range(0, X.shape[0]):
				if np.isnan(X[j,i]):
					X[j,i] = X[j-1,i]
		cut = int(len(X)*0.5)  # done to preserve anomaly rate in train and test 
		X_train = X[:cut]
		X_test = X[cut:]
		y_train = y[:cut]
		y_test = y[cut:]
		# take out anomalies of train data (for GECCO_normal)
		if dataset == 'GECCO_normal':
			idx = np.where(y_train == 0)
			X_train = X_train[idx]
			y_train = y_train[idx]
		y_train = y_train[:, np.newaxis]
		y_test = y_test[:, np.newaxis]
		# clip data to [2, 98] percentile of train data (except for first feature)
		for dim in range(9):
			if dim > 0:
				x_up = np.percentile(X_train[:,dim], 98)
				x_low = np.percentile(X_train[:,dim], 2)
				X_train[:,dim] = np.clip(X_train[:,dim], x_low, x_up)
				X_test[:,dim] = np.clip(X_test[:,dim], x_low, x_up)
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)
		print(X_train_scaled.shape, X_test_scaled.shape, y_test.shape)
		np.save(f'{output_folder}/{dataset}/train.npy', X_train_scaled)
		np.save(f'{output_folder}/{dataset}/test.npy', X_test_scaled)
		np.save(f'{output_folder}/{dataset}/labels.npy', y_test)
	elif dataset == 'IEEECIS':   # from fraud-dataset-benchmark
		# basic idea is to extract time series for each user with more than 50 transactions for train
		# and more than 20 transactions for test 
		x_train = pd.read_csv(f'{data_folder}/ieeecis/train.csv')
		x_test = pd.read_csv('data/ieeecis/test.csv')
		labels = pd.read_csv('data/ieeecis/labels.csv')

		non_num_cols = x_train.select_dtypes(exclude=['float', 'int']).columns
		x_train = x_train.sort_values(['EVENT_TIMESTAMP'])
		x_test = x_test.sort_values(['EVENT_TIMESTAMP'])
		feature_names = list(x_train.columns)

		# Group by ENTITY_ID for train data
		train_group = x_train.groupby('ENTITY_ID')
		# choose user time series in group that are >= 40
		train_uid = train_group.size()[train_group.size() >= 40].index
		train_uid = list(train_uid)
		# get the transaction ids for each user which has more than 50 transactions
		train_tranID = np.empty((0))
		for _, group in train_group:
			if len(group) >= 50:
				train_tranID = np.concatenate((train_tranID, group['TransactionID'].values), axis=None) 
		train_tranID.flatten()  # don't care which uid they belond to, just want list of used transactions

		# repeat same for testing, but accept shorter time series (i.e. >=20)
		test_group = x_test.groupby('ENTITY_ID')
		test_uid = test_group.size()[test_group.size() >= 20].index  # allow shorter time series for testing!
		test_uid = list(test_uid)
		# get the transaction ids for each user which has more than 20 transactions for testing
		test_tranID = np.empty((0))
		for _, group in test_group:
			if len(group) >= 20:
				test_tranID = np.concatenate((test_tranID, group['TransactionID'].values), axis=None) 
		test_tranID.flatten()  # don't care which uid they belond to, just want list of used transactions

		# get train data for chosen transactions in train_tranID
		x_train = x_train[x_train['TransactionID'].isin(train_tranID)]  
		# get test data for chosen transactions in test_tranID
		x_test = x_test[x_test['TransactionID'].isin(test_tranID)]

		# one hot encoding for categorical features in train data
		encoding = {}       # for categorical features, dict containing encode & feature name as key
		other_values = {}   # for categorical features, dict containing less frequent categories that will just be replaced by 'other' 
		x_train1 = x_train.copy()  # to avoid modifying initial input data
		for i, feat in enumerate(feature_names):
			print(feat)
			if feat in non_num_cols and feat not in ['ENTITY_ID', 'EVENT_TIMESTAMP']:  # fill nan values + do encoding for cat features (except for uid and timestamp)
				print('cat feature: ', feat)
				x_train1[feat] = x_train1[feat].fillna('missing')
				arr = x_train1[feat].values
				unique_values, counts = np.unique(arr, return_counts=True)
				if len(unique_values) > 50:  # take 100 most frequent values for encoding
					idx = np.argsort(counts)
					unique_values = unique_values[idx]  # unique values sorted in frequency
					other_values[feat] = unique_values[49:]   # values to be encoded as 'other'
					# replace all elements of x_train1[feat] that are in other_values with 'other'
					idx = np.where(np.isin(arr, other_values[feat]))
					arr[idx] = 'other'
				enc = OneHotEncoder(handle_unknown='ignore')
				encoded_arr = enc.fit_transform(arr.reshape(-1,1)).toarray()
				encoding[feat] = enc    # save encoder for testing data
				new_feat = pd.DataFrame(encoded_arr, columns=enc.get_feature_names_out(input_features =[feat]))
				new_feat = new_feat.reset_index(drop=True)
				x_train1.drop(columns=[feat], inplace=True)
				x_train1 = x_train1.reset_index(drop=True)   # Reset index to ensure proper concatenation with new_feat (where indexing starts from 0)
				x_train1 = pd.concat([x_train1, new_feat], axis=1)
				print('checking if there are any nan entries:\n', 
					x_train1.loc[:,enc.get_feature_names_out(input_features =[feat])].isna().sum())
			else:  # fill nan values
				print('num feature: ', feat)
				x_train1.loc[:,feat] = x_train1.loc[:,feat].fillna(0)

				print('checking if there are any nan entries:', x_train1.loc[:,feat].isna().sum())

		# one hot encoding for categorical features in test data
		x_test1 = x_test.copy()  # to avoid modifying initial input data
		for i, feat in enumerate(feature_names):
			print(feat)
			if feat in non_num_cols and feat not in ['ENTITY_ID', 'EVENT_TIMESTAMP']:  # fill nan values + do encoding for cat features (except for uid and timestamp)
				print('cat feature: ', feat)
				x_test1[feat] = x_test1[feat].fillna('missing')
				arr = x_test1[feat].values
				if feat in other_values:
					# replace all elements of x_test1[feat] that are in other_values (previously defined) with 'other'
					idx = np.where(np.isin(arr, other_values[feat]))
					arr[idx] = 'other'
				enc = encoding[feat]  # use the previously fitted encoder
				encoded_arr = enc.transform(arr.reshape(-1,1)).toarray()
				new_feat = pd.DataFrame(encoded_arr, columns=enc.get_feature_names_out(input_features =[feat]))
				new_feat = new_feat.reset_index(drop=True)
				x_test1.drop(columns=[feat], inplace=True)
				x_test1 = x_test1.reset_index(drop=True)   # Reset index to ensure proper concatenation with new_feat (where indexing starts from 0)
				x_test1 = pd.concat([x_test1, new_feat], axis=1)				
				print('checking if there are any nan entries:\n', 
					x_test1.loc[:,enc.get_feature_names_out(input_features =[feat])].isna().sum())
			else:  # fill nan values
				print('num feature: ', feat)
				x_test1.loc[:,feat] = x_test1.loc[:,feat].fillna(0)
				print('checking if there are any nan entries:', x_test1.loc[:,feat].isna().sum())

		# update feature names and drop some columns before scaling
		updated_feature_names = list(x_train1.columns)
		train_info = x_train1['ENTITY_ID']
		train_info.index = x_train['TransactionID']
		test_info = x_test1['ENTITY_ID']
		test_info.index = x_test['TransactionID']
		x_train1.drop(columns=['ENTITY_ID', 'TransactionID', 'EVENT_TIMESTAMP'], inplace=True)
		x_test1.drop(columns=['ENTITY_ID', 'TransactionID', 'EVENT_TIMESTAMP'], inplace=True)
		# scaling
		scaler = StandardScaler()
		x_train1_scaled = scaler.fit_transform(x_train1)
		x_test1_scaled = scaler.transform(x_test1)
		# add info column back
		updated_feature_names = list(x_train1.columns)
		df_train = pd.DataFrame(x_train1_scaled, index=train_info.index, columns=updated_feature_names)
		df_test = pd.DataFrame(x_test1_scaled, index=test_info.index, columns=updated_feature_names)
		df_train = pd.concat([df_train, train_info], axis=1)
		df_test = pd.concat([df_test, test_info], axis=1)
		df_train.sort_index(axis=0, inplace=True)
		df_test.sort_index(axis=0, inplace=True)
		test_tranID = np.array(test_info.index)

		# regroup to access previously selected train TS if NOT applying PCA
		train_group2 = df_train.groupby('ENTITY_ID')
		for uid, group in train_group2:
			arr = group.values[:, :-1]  # drop the ENTITY_ID column
			arr = np.array(arr, dtype=np.float64)
			print(f'time series for user: {uid} has shape:\n {arr.shape}')
			np.save(f'{output_folder}/{dataset}/train_{uid}.npy', arr)
		# regroup to access previously selected test TS
		test_group2 = df_test.groupby('ENTITY_ID')
		anomaly_count = 0
		for uid, group in test_group2:
			# print(len(group.index))
			arr = group.values[:, :-1]  # drop the ENTITY_ID column
			arr = np.array(arr, dtype=np.float64)
			# extract labels of the transactions we actually use in TS
			label_idx = np.where(np.isin(labels['TransactionID'], list(group.index)))
			y_test = np.array(labels.loc[label_idx, 'EVENT_LABEL'])
			y_test = y_test[:, np.newaxis]
			anomaly_count += len(y_test[y_test == 1])
			np.save(f'{output_folder}/{dataset}/test_{uid}.npy', arr)
			np.save(f'{output_folder}/{dataset}/labels_{uid}.npy', y_test)
		print('number of anomalies in test data:', anomaly_count)

	elif dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset == 'UCR':
		dataset_folder = 'data/UCR'
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.txt'): continue
			vals = filename.split('.')[0].split('_')
			dnum, vals = int(vals[0]), vals[-3:]
			vals = [int(i) for i in vals]
			temp = np.genfromtxt(os.path.join(dataset_folder, filename),
								dtype=np.float64,
								delimiter=',')
			min_temp, max_temp = np.min(temp), np.max(temp)
			temp = (temp - min_temp) / (max_temp - min_temp)
			train, test = temp[:vals[0]], temp[vals[0]:]
			labels = np.zeros_like(test)
			labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = 'data/SMAP_MSL'
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'SWaT_1D':
		dataset_folder = 'data/SWaT_1D'
		file = os.path.join(dataset_folder, 'series.json')
		df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
		df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
		train, min_a, max_a = normalize2(df_train.values)
		test, _, _ = normalize2(df_test.values, min_a, max_a)
		labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SWaT':
		dataset_folder = 'data/SWaT'
		train = pd.read_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v1.csv'))
		test = pd.read_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.csv'))
		train.columns = train.iloc[0] # set column names as first row
		train = train.drop(train.index[0]) # drop first row
		test.columns = test.iloc[0] # set column names as first row
		test = test.drop(test.index[0]) # drop first row
		train.columns = train.columns.str.strip()
		test.columns = test.columns.str.strip()
	
		train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		train.fillna(0, inplace=True); test.fillna(0, inplace=True)

		test['Normal/Attack'] = test['Normal/Attack'].str.replace(" ", "")
		labels = test['Normal/Attack']
		
		labels = (labels == 'Attack').astype(int)
		
		train = train.drop(columns=['Normal/Attack', 'Timestamp']).astype(float)  # are all 'Normal' anyways
		test = test.drop(columns=['Normal/Attack', 'Timestamp']).astype(float)
		train, test = convertNumpy(train, reduce=True), convertNumpy(test, reduce=True) # downsampling, only keep 1/10 of data
		labels = labels[::10]  # downsampling, only keep 1/10 of data
		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'WADI':
		dataset_folder = 'data/WADI/WADI.A2_19Nov2019'  # use version A2 where unstable period is removed
		train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days_new.csv'))
		test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdataLABLE.csv'))
		train.drop(columns=['Row', 'Time', 'Date']), test.drop(columns=['Row', 'Time', 'Date'])
		train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		train.fillna(0, inplace=True); test.fillna(0, inplace=True)
		labels = test["AttackLABLE (1:No Attack, -1:Attack)"]
		test = test.drop(columns=["AttackLABLE (1:No Attack, -1:Attack)"])
		train, test = convertNumpy(train, reduce=False), convertNumpy(test, reduce=False) # downsampling, only keep 1/10 of data
		labels = labels.values  # 1 for normal, -1 for attack, but want 0 for normal, 1 for attack
		labels = (1 - labels) / 2
		#labels = labels[::10]  # downsampling, only keep 1/10 of data
		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")