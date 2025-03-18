import os, sys, math
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from time import time
from pprint import pprint
from torchinfo import summary

from src.constants import *
from src.plotting import plot_accuracies, plot_labels, plot_losses, plotter, plotter2, compare_labels
# from src.pot import SPOT
from src.dlutils import ComputeLoss
from src.utils import load_model, save_model, EarlyStopper
from src.diagnosis import hit_att, ndcg
from src.merlin import *
from src.utils import combined_loss
from src.data_loader import MyDataset


def backprop(epoch, model, data, feats, optimizer, scheduler, training=True, enc_feats=0, prob=False, pred=False):
	
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.window_size
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.window_size
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			loss = l(y_pred, data)
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.window_size
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.window_size
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()
	elif 'GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.window_size
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		# data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		# bs = model.batch if training else len(data)
		# dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1
		l1s, l2s = [], []
		if training:
			for d in data:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				# if not l1s and n<=1: 
				# 	summary(model, input_data=[window, elem])
				z = model(window, elem)
				if prob:  # sample from probabilistic output
					if isinstance(z, tuple):  # if z = (x1, x2)
						x1 = z[0]; x2 = z[1]
						x1_mu, x1_logsigma = torch.split(x1, split_size_or_sections=feats, dim=2)
						x1 = x1_mu + torch.randn(size=x1_logsigma.size()) * torch.exp(x1_logsigma)
						x2_mu, x2_logsigma = torch.split(x2, split_size_or_sections=feats, dim=2)
						x2 = x2_mu + torch.randn(size=x2_logsigma.size()) * torch.exp(x2_logsigma)
						z = (x1, x2)
					else:  # if z = x2
						z_mu, z_logsigma = torch.split(z, split_size_or_sections=feats, dim=2)
						z = z_mu + torch.randn(size=z_logsigma.size())*torch.exp(z_logsigma)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			loss = torch.empty(0)
			z_all = torch.empty(0)
			for d in data:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				if prob:  # don't sample from probabilistic output for testing, just use mean
					if isinstance(z, tuple):  # if z = (x1, x2)
						x1 = z[0]; x2 = z[1]
						x1_mu, x1_logsigma = torch.split(x1, split_size_or_sections=feats, dim=2)
						x2_mu, x2_logsigma = torch.split(x2, split_size_or_sections=feats, dim=2)
						z = (x1_mu, x2_mu)
					else:  # if z = x2
						z_mu, z_logsigma = torch.split(x1, split_size_or_sections=feats, dim=2)
						z = z_mu
				if isinstance(z, tuple): z = z[1]
				l1 = l(z, elem)[0]
				loss = torch.cat((loss, l1.view(-1, feats)), dim=0)
				if pred: z_all = torch.cat((z_all, z.view(-1, feats)), dim=0)
			if pred:
				return loss.detach().numpy(), z_all.detach().numpy()
			else:
				return loss.detach().numpy()
	elif 'iTransformer' in model.name:
		# l = combined_loss(model.window_size)
		l = nn.MSELoss(reduction = 'none')
		# l = nn.HuberLoss(reduction = 'none')
		n = epoch + 1
		if model.weighted:
			mid = model.window_size % 2
			middle = math.floor(model.window_size / 2)
			weights = [i + 1 for i in range(middle)] + mid * [middle+1] + [i for i in range(middle, 0, -1)]
			weights /= np.sum(weights)
			weights = torch.tensor(weights).view(-1,1).double()
		if training:
			l1s = []
			for d in data: # d.shape is [B, window_size, N]
				local_bs = d.shape[0]
				if enc_feats > 0:
					d_enc = d[:, :, :enc_feats]
					d = d[:, :, enc_feats:]
				else:
					d_enc = None
				# don't invert d because we have permutation later in DataEmbedding_inverted as part of model
				if model.output_attention:
					z = model(d, d_enc)[0]
				else:
					z = model(d, d_enc)
				if prob:  # sample from probabilistic output
					z_mu = z[0]
					z_logsigma = z[1]
					z = z_mu + torch.randn(size=z_logsigma.size())*torch.exp(z_logsigma)
				l1 = l(z, d)
				# if model.weighted:
				# 	l1 = l1 * weights
				l1s.append(torch.mean(l1).item())
				if prob:
					z_std = torch.exp(z_logsigma)
					loss = torch.mean(l1/z_std) + torch.mean(z_std)
					# loss = torch.mean(l1)
					# loss_fct = torch.nn.GaussianNLLLoss(eps=1e-6, reduction='mean')
					# loss = loss_fct(z, elem, torch.exp(2*z_logsigma))
				else:
					loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			z_all = torch.empty(0)
			if model.weighted:
				# loss = torch.zeros(size=(model.batch * model.window_size, feats))
				# new test: get loss according to position in window
				loss = torch.zeros(size=(model.window_size, feats))
			else:
				loss = torch.empty(0)
			for i, d in enumerate(data): # d.shape is [B, window_size, N]
				local_bs = d.shape[0]
				if enc_feats > 0:
					d_enc = d[:, :, :enc_feats]
					d = d[:, :, enc_feats:]
				else:
					d_enc = None
				if model.output_attention:
					z = model(d, d_enc)[0]
				else:
					z = model(d, d_enc)
				if prob:  # don't sample from probabilistic output for testing, just use mean
					z_mu = z[0]
					z_logsigma = z[1]
					z = z_mu
				l1 = l(z, d)
				# z_out, z25, z75 = torch.split(z, model.window_size, dim=1)  # when using combined_loss
				# l1 = l(z_out, d)
				# l1 = ((z_out - d) * (z75 - z25))**2
				# l1 = torch.abs(l(z, d))
				# loss_fct = torch.nn.GaussianNLLLoss(eps=1e-6, reduction='none')
				# l1 = loss_fct(z, elem, torch.exp(2*z_logsigma))
				if model.weighted:
					# l1_weighted = l1 * weights 
					# idx = i * local_bs * model.test_step_size
					# for j in range(local_bs):
					# 	start = idx +  j*model.test_step_size
					# 	stop = start + model.window_size	
					# 	loss[start:stop] += l1_weighted[j]
					# print(start, stop, idx)
					# new test: get loss according to position in window
					l1s = l1.mean(dim=0)
					loss += l1s
				else:
					loss = torch.cat((loss, l1), dim=0)
					# loss = torch.cat((loss, l1.reshape(-1, feats)), dim=0)
				if pred:
					z_all = torch.cat((z_all, z), dim=0)
					# z_all = torch.cat((z_all, z.reshape(-1, feats)), dim=0)
			if not model.weighted:
				loss = loss.view(-1, feats)
			# if prob:
			# 	z_std = torch.exp(z_logsigma)
			# 	loss = loss / z_std  #+ z_std
			if pred:
				# z_all, z25, z75 = torch.split(z_all, model.window_size, dim=1)
				z_all = z_all.reshape(-1, feats)
				# z25 = z25.reshape(-1, feats)
				# z75 = z75.reshape(-1, feats)
				return loss.detach().numpy(), z_all.detach().numpy()  #, z25.detach().numpy(), z75.detach().numpy() # because we have unnecessary third dimension
			else:
				return loss.detach().numpy()
	elif 'LSTM_AE' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bs = model.batch # if training else len(data)
		n = epoch + 1
		if training:
			l1s = []
			for d in data:
				y_pred = model(d)
				l1 = l(y_pred, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			# tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			loss = torch.empty(0)
			z_all = torch.empty(0)
			for d in data:
				y_pred = model(d)
				l1 = l(y_pred, d)
				loss = torch.cat((loss, l1), dim=0)
				z_all = torch.cat((z_all, d), dim=0)
			loss = loss.view((-1, feats))
			z_all = z_all.view((-1, feats))
			if pred:
				return loss.detach().numpy(), z_all.detach().numpy()
			else:
				return loss.detach().numpy()
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			if pred:
				return loss.detach().numpy(), y_pred.detach().numpy()
			else:
				return loss.detach().numpy()


def local_pot(loss, lossT, labels, q=1e-5, plot_path=None):
	# get anomaly labels
	df_res_local = pd.DataFrame()
	preds = []
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]  	
		result_local, pred = pot_eval(lt, l, ls, plot_path, f'dim{i}', q=q)
		preds.append(pred)
		df_res = pd.DataFrame.from_dict(result_local, orient='index').T
		df_res_local = pd.concat([df_res_local, df_res], ignore_index=True)
	preds = np.array(preds).T
	preds = preds.astype(int)
	return preds, df_res_local


def local_anomaly_labels(preds, labels, q=1e-5, plot_path=None, nb_adim=1):
	labelspred = (np.sum(preds, axis=1) >= nb_adim) + 0

	if plot_path is not None:
		plot_labels(plot_path, f'labels_adim{nb_adim}', y_pred=labelspred, y_true=labels)
	
	result_local = calc_point2point(predict=labelspred, actual=labels)
	result_local1 = {'f1': result_local[0], 'precision': result_local[1], 'recall': result_local[2], 
				'TP': result_local[3], 'TN': result_local[4], 'FP': result_local[5], 'FN': result_local[6], 
				'ROC/AUC': result_local[7], 'MCC': result_local[8]}
	result_local1.update({'detection level q': q})
	print(f'local results with {nb_adim} anomalous dimensions for anomaly')
	pprint(result_local1)

	return labelspred, result_local1


if __name__ == '__main__':
	print(args, '\n')
	print(torch.cuda.is_available())

	# define path for results, checkpoints & plots & create directories
	if args.name:
		folder = f'{args.model}_param_search/{args.model}_{args.dataset}/window{args.window_size}_steps{args.step_size}_dmodel{args.d_model}_feats{args.feats}_eps{args.epochs}/{args.name}'
	else:
		folder = f'{args.model}_param_search/{args.model}_{args.dataset}/window{args.window_size}_steps{args.step_size}_dmodel{args.d_model}_feats{args.feats}_eps{args.epochs}'
	plot_path = f'{folder}/plots'
	res_path = f'{folder}/results'
	if args.checkpoint is None:
		checkpoints_path = f'{folder}/checkpoints'
	else:
		checkpoints_path = args.checkpoint
	os.makedirs(plot_path, exist_ok=True)
	os.makedirs(res_path, exist_ok=True)

	train = MyDataset(args.dataset, args.window_size, args.step_size, args.model, flag='train', feats=args.feats, less=args.less, enc=args.enc, k=args.k)
	if args.weighted:
		test = MyDataset(args.dataset, args.window_size, args.step_size, args.model, flag='test', feats=args.feats, less=args.less, enc=args.enc, k=-1)
		train_test = MyDataset(args.dataset, args.window_size, args.step_size, args.model, flag='train', feats=args.feats, less=args.less, enc=args.enc, k=-1)
	else:
		test = MyDataset(args.dataset, args.window_size, args.window_size, args.model, flag='test', feats=args.feats, less=args.less, enc=args.enc, k=-1)
		train_test = MyDataset(args.dataset, args.window_size, args.window_size, args.model, flag='train', feats=args.feats, less=args.less, enc=args.enc, k=-1)
	labels = test.get_labels()
	
	print('train set', train.__len__(), train.data.shape)
	print('test set', test.__len__(), test.data.shape)
	print('labels', labels.shape)
	
	if args.k > 0:
		valid = MyDataset(args.dataset, args.window_size, args.step_size, args.model, flag='valid', feats=args.feats, less=args.less, enc=args.enc, k=args.k)
		print(f'{args.k}-fold valid set', valid.__len__(), valid.data.shape)

	feats = train.feats
	enc_feats = train.enc_feats
	
	if args.model in ['MERLIN']:
		data_loader_test = DataLoader(test, batch_size=24, shuffle=False)
		eval(f'run_{args.model.lower()}(data_loader_test, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = \
		load_model(args.model, args.dataset, feats, args.window_size, args.step_size, args.d_model, args.test, checkpoints_path, args.prob, args.weighted)

	# Calculate and print the number of parameters
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'total params: {total_params}, trainable params: {trainable_params}')

	# Create data loader
	data_loader_train = DataLoader(train, batch_size=model.batch, shuffle=False)
	if args.model not in ['MERLIN']:
		data_loader_test = DataLoader(test, batch_size=model.batch, shuffle=False)
	data_loader_train_test = DataLoader(train_test, batch_size=model.batch, shuffle=False)
	if args.k > 0:
		data_loader_valid = DataLoader(valid, batch_size=model.batch, shuffle=False)

	# save arguments and additional info in config file
	with open(f'{folder}/config.txt', 'w') as f:
		f.write(f'{args.model} on {args.dataset}\n \n')
		f.write(str(args)+'\n')
		f.write(f'total params: {total_params}, trainable params: {trainable_params}\n')
		f.write(f'feats: {feats}\n')
		f.write(f'train: {train.__len__()}\n')
		if args.k > 0:
			f.write(f'valid: {valid.__len__()}\n')
		f.write(f'test: {test.__len__()}\n')
		f.write(f'ts_lengths train: {train.get_ts_lengths()}\n')
		if args.k > 0:
			f.write(f'ts_lengths valid: {valid.get_ts_lengths()}\n')
		f.write(f'ts_lengths test: {test.get_ts_lengths()}\n')
		f.write(f'optimizer: {optimizer}, \nscheduler: {scheduler}\n')

		sample_batch = next(iter(data_loader_train))
		f.write('\nModel Summary:\n')
		if model.name == 'TranAD':
			window = sample_batch.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, -1, feats)
			f.write(str(summary(model, input_data=[window, elem], depth=5, verbose=0)))
		else:
			f.write(str(summary(model, input_data=sample_batch, depth=5, verbose=0)))
		
	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		if args.k > 0:
			early_stopper = EarlyStopper(patience=5, min_delta=0.0)
			min_lossV = 100
		num_epochs = args.epochs; e = epoch + 1; start_time = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, data_loader_train, feats, optimizer, scheduler, training=True, enc_feats=enc_feats, prob=args.prob)
			if args.k > 0:
				lossV = backprop(e, model, data_loader_valid, feats, optimizer, scheduler, training=False, enc_feats=enc_feats, prob=args.prob)
				lossV = np.mean(lossV)
				if lossV < min_lossV:
					min_lossV = lossV
					save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list, '_best')
			else:
				lossV = 0
			tqdm.write(f'Epoch {e},\tL_train = {lossT}, \t\tL_valid = {lossV}, \tLR = {lr}')
			accuracy_list.append((lossT, lossV, lr))
			save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list, f'_epoch{e}')
			if args.k > 0 and early_stopper.early_stop(-lossV):
				print(f'{color.HEADER}Early stopping at epoch {e}{color.ENDC}')
				break
		train_time = time() - start_time
		print(f'{color.BOLD}Training time: {"{:10.4f}".format(train_time)} s or {"{:.2f}".format(train_time/60)} min {color.ENDC}')
		save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list, '_final')
		if not os.path.exists(f'{checkpoints_path}/model_best.ckpt'):
			save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list, '_best')
		plot_accuracies(accuracy_list, plot_path)
		plot_losses(accuracy_list, plot_path)
		np.save(f'{checkpoints_path}/accuracy_list.npy', accuracy_list)

	### Testing phase
	torch.zero_grad = True
	if args.k > 0:  # if using validation set, make sure to load best model
		checkpoint = torch.load(f'{checkpoints_path}/model_best.ckpt')
		model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

	### Scores
	lossT = backprop(-1, model, data_loader_train_test, feats, optimizer, scheduler, training=False, enc_feats=enc_feats, prob=args.prob, pred=False)  # need anomaly scores on training data for POT
	# loss, y_pred, y25, y75 = backprop(-1, model, data_loader_test, feats, optimizer, scheduler, training=False, enc_feats=enc_feats, prob=args.prob, pred=True)	
	loss, y_pred = backprop(-1, model, data_loader_test, feats, optimizer, scheduler, training=False, enc_feats=enc_feats, prob=args.prob, pred=True)

	# just for studies_posinwindow
	# fig, axs = plt.subplots(nrows=feats, ncols=1, figsize=(16, feats * 2))
	# for dim in range(feats):
	# 	if feats == 1:
	# 		ax = axs
	# 	else:
	# 		ax = axs[dim]
	# 	ax.plot(lossT[:, dim], '-o', label='train loss')
	# 	ax.plot(loss[:, dim], '-o', label='test loss')
	# 	if args.window_size > 20:
	# 		ax.set_xticks(np.arange(0, args.window_size, 10))
	# 	else:
	# 		ax.set_xticks(np.arange(0, args.window_size))
	# 	ax.set_ylabel(f'Dim {dim}', rotation=0, ha='right', rotation_mode='default', labelpad=10)
	# 	ax.legend(loc='upper right')
	# if feats == 1:
	# 	axs.set_xlabel('Position in time window')
	# else:
	# 	axs[-1].set_xlabel('Position in time window')
	# fig.subplots_adjust(hspace=0.2)  
	# fig.align_ylabels(axs)
	# plt.tight_layout()
	# plt.savefig(f'studies_posinwindow/{args.model}_{args.dataset}_loss_posinw_window{args.window_size}.png', dpi=300)
	# plt.close()

	# # Plot average train and test loss over all dimensions
	# plt.figure(figsize=(10, 6))
	# plt.plot(np.mean(lossT, axis=1), '-o', label='Average train loss')
	# plt.plot(np.mean(loss, axis=1), '-o', label='Average test loss')
	# plt.xlabel('Position in time window')
	# plt.ylabel('Average loss across features')
	# plt.legend(loc='upper right')
	# plt.tight_layout()
	# plt.savefig(f'studies_posinwindow/{args.model}_{args.dataset}_avgloss_posinw_window{args.window_size}.png', dpi=300)
	# plt.close()
	# print('plot saved')
	# sys.exit()

	if feats <= 30:
		testOO = test.get_complete_data_wpadding()
		nolabels = np.zeros_like(loss)
		print(len(testOO), len(y_pred), len(loss))
		plotter(plot_path, testOO, y_pred, loss, nolabels, test.get_ideal_lengths(), name='output_padded')
	
	print(lossT.shape, loss.shape, labels.shape)
	if 'iTransformer' in model.name or model.name in ['LSTM_AE']:
		# cut out the padding from test data, loss tensors
		lossT_tmp, loss_tmp, y_pred_tmp = [], [], []
		y25_tmp, y75_tmp = [], []
		print(test.get_ts_lengths(), np.sum(test.get_ts_lengths()), len(test.get_ts_lengths()))
		print(test.get_ideal_lengths(), np.sum(test.get_ideal_lengths()), len(test.get_ideal_lengths()))
		start = 0
		for i, l in enumerate(test.get_ts_lengths()):
			loss_tmp.append(loss[start:start+l])
			y_pred_tmp.append(y_pred[start:start+l])
			# y25_tmp.append(y25[start:start+l])
			# y75_tmp.append(y75[start:start+l])
			start += test.get_ideal_lengths()[i]
		
		start = 0
		for i, l in enumerate(train_test.get_ts_lengths()):
			lossT_tmp.append(lossT[start:start+l])
			start += train_test.get_ideal_lengths()[i]

		lossT = np.concatenate(lossT_tmp, axis=0)
		loss = np.concatenate(loss_tmp, axis=0)
		y_pred = np.concatenate(y_pred_tmp, axis=0)
		# y25 = np.concatenate(y25_tmp, axis=0)
		# y75 = np.concatenate(y75_tmp, axis=0)
	print(lossT.shape, loss.shape, labels.shape)
	train_loss = np.mean(lossT)
	test_loss = np.mean(loss)

	### Plot curves
	if feats <= 40:
		testO = test.get_complete_data()
		print(len(testO), len(y_pred), len(labels))
		plotter(plot_path, testO, y_pred, loss, labels, test.get_ts_lengths(), name='output')

	# # if step_size > 1, define truth labels per window instead of per time stamp, can also just use non-overlapping windows for testing
	# if args.step_size > 1:
	# 	print(labels.shape, labels[labels[:,0]==1].shape, labels[labels[:,0]==0].shape)
	# 	nb_windows = (len(labels) - args.window_size) // args.step_size
	# 	labels = np.array([(np.sum(labels[i*args.step_size:i*args.step_size+args.window_size], axis=0) >= 1) + 0 for i in range(nb_windows)])
	# 	print(labels.shape, labels[labels[:,0]==1].shape, labels[labels[:,0]==0].shape)

	### anomaly labels
	preds, df_res_local = local_pot(loss, lossT, labels, args.q, plot_path)
	true_labels = (np.sum(labels, axis=1) >= 1) + 0
	# local anomaly labels
	labelspred, result_local1 = local_anomaly_labels(preds, true_labels, args.q, plot_path, nb_adim=1)
	majority = math.ceil(labels.shape[1] / 2)	# do majority voting over dimensions for local results instead of inclusive OR
	labelspred_maj, result_local2 = local_anomaly_labels(preds, true_labels, args.q, plot_path, nb_adim=majority)
	labelspred_all = []
	results_all = pd.DataFrame()

	if feats > 20:
		nb_adim = np.unique(np.concatenate((np.arange(1, 11), np.arange(20, feats, 10), [feats])))
	else:
		nb_adim = np.arange(1, feats+1)
	for i in nb_adim:
		lpred, res = local_anomaly_labels(preds, true_labels, args.q, plot_path, nb_adim=i)
		labelspred_all.append(lpred)
		results_all = pd.concat([results_all, pd.DataFrame.from_dict(res, orient='index').T], ignore_index=True)
	
	results_all.index = nb_adim
	results_all.to_csv(f'{res_path}/res_local_all.csv')

	# global anomaly labels
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	true_labels = (np.sum(labels, axis=1) >= 1) + 0
	result_global, pred2 = pot_eval(lossTfinal, lossFinal, true_labels, plot_path, f'all_dim', q=args.q)
	labelspred_glob = (pred2 >= 1) + 0

	plot_labels(plot_path, 'labels_global', y_pred=labelspred_glob, y_true=true_labels)
	# metrics_global = calc_point2point(predict=labelspred_glob, actual=true_labels)
	result_global.update(hit_att(loss, labels))
	result_global.update(ndcg(loss, labels))
	if not args.test:
		result_global.update({'train_time': train_time})
	result_global.update({'detection_level_q': args.q})
	result_global.update({'train_loss': train_loss, 'test_loss': test_loss})
	print('\nglobal results') 
	pprint(result_global)

	plot_metrics(plot_path, ['local (incl. OR)', 'local (maj. voting)', 'global'], 
			  y_pred=[labelspred, labelspred_maj, labelspred_glob], y_true=true_labels)

	# compare local & global anomaly labels
	compare_labels(plot_path, pred_labels=[labelspred, labelspred_maj, labelspred_glob], true_labels=true_labels, 
				plot_labels=['Local anomaly\n(inclusive OR)', 'Local anomaly\n(majority voting)', 'Global anomaly'], 
				name='_all')
	
	if feats <= 40:
		plotter2(plot_path, testO, y_pred, loss, args.dataset, labelspred, labels, name='_local_or')
		plotter2(plot_path, testO, y_pred, loss, args.dataset, labelspred_maj, labels, name='_local_maj')
		plotter2(plot_path, testO, y_pred, loss, args.dataset, labelspred_glob, labels, name='_global')
		# plotter2(plot_path, testO, [y_pred, y25, y75], loss, args.dataset, labelspred, labels, name='_local_or')
		# plotter2(plot_path, testO, [y_pred, y25, y75], loss, args.dataset, labelspred_maj, labels, name='_local_maj')
		# plotter2(plot_path, testO, [y_pred, y25, y75], loss, args.dataset, labelspred_glob, labels, name='_global')

	# saving results
	df_res_global = pd.DataFrame.from_dict(result_global, orient='index').T
	df_res_global.index = ['global']
	result_local1 = pd.DataFrame.from_dict(result_local1, orient='index').T
	result_local2 = pd.DataFrame.from_dict(result_local2, orient='index').T
	result_local1.index = ['local_all']
	result_local2.index = ['local_all_maj']
	df_res_local = pd.concat([df_res_local, result_local1, result_local2])
	df_res = pd.concat([df_res_local, df_res_global]) 
	df_labels = pd.DataFrame( {'local_or': labelspred, 'local_maj': labelspred_maj, 'global': labelspred_glob} )

	df_res.to_csv(f'{res_path}/res.csv')	
	df_labels.to_csv(f'{res_path}/pred_labels.csv', index=False)
