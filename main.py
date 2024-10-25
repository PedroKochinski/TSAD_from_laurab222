import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
from torchinfo import summary

from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import load_model, save_model
from src.diagnosis import *
from src.merlin import *
from src.data_loader import load_dataset, convert_to_windows_train, convert_to_windows_test


def backprop(epoch, model, data, feats, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
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
			return loss.detach().numpy(), y_pred.detach().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
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
			return loss.detach().numpy(), y_pred.detach().numpy()
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
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
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
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
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
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.n_window
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
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
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
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				# if not l1s and n<=1: 
				# 	summary(model, input_data=[window, elem])
				z = model(window, elem)
				if args.prob:  # sample from probabilistic output
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
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if args.prob:  # don't sample from probabilistic output for testing, just use mean
					if isinstance(z, tuple):  # if z = (x1, x2)
						x1 = z[0]; x2 = z[1]
						x1_mu, x1_logsigma = torch.split(x1, split_size_or_sections=feats, dim=2)
						x2_mu, x2_logsigma = torch.split(x2, split_size_or_sections=feats, dim=2)
						z = (x1_mu, x2_mu)
					else:  # if z = x2
						z_mu, z_logsigma = torch.split(x1, split_size_or_sections=feats, dim=2)
						z = z_mu
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	elif 'iTransformer' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data)
		dataset = TensorDataset(data_x, data_x)
		bs = model.batch # if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1
		if training:
			l1s = []
			for d, _ in dataloader:
				local_bs = d.shape[0]
				# don't invert d because we have permutation later in DataEmbedding_inverted as part of model
				# elem = d.permute(1, 0, 2)[-1, :, :].view(1, local_bs, feats)  # [1, B, N]
				elem = d.permute(1, 0, 2) # [n_window, B, N]
				# if not l1s: 
				# 	summary(model, input_size=[1, 10, 25])
				if model.output_attention:
					z = model(d)[0]
				else:
					z = model(d)
				if args.prob:  # sample from probabilistic output
					z_mu = z[0]
					z_logsigma = z[1]
					z = z_mu + torch.randn(size=z_logsigma.size())*torch.exp(z_logsigma)
				l1 = l(z, elem)
				l1s.append(torch.mean(l1).item())
				if args.prob:
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
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			loss = torch.empty(0)
			z_all = torch.empty(0)
			for d, _ in dataloader:
				local_bs = d.shape[0]
				# don't invert d because we have permutation later in DataEmbedding_inverted
				# elem = d.permute(1, 0, 2)[-1, :, :].view(1, local_bs, feats)
				elem = d.permute(1, 0, 2) # [n_window, B, N]
				if model.output_attention:
					z = model(d)[0]
				else:
					z = model(d)
				if args.prob:  # don't sample from probabilistic output for testing, just use mean
					z_mu = z[0]
					z_logsigma = z[1]
					z = z_mu
				l1 = l(z, elem)
				# loss_fct = torch.nn.GaussianNLLLoss(eps=1e-6, reduction='none')
				# l1 = loss_fct(z, elem, torch.exp(2*z_logsigma))
				loss = torch.cat((loss, l1.permute(1, 0, 2)), dim=0)
				z_all = torch.cat((z_all, z.permute(1, 0, 2)), dim=0)
			loss = loss.view((data.shape[0]* data.shape[1], feats))
			z_all = z_all.view((data.shape[0]* data.shape[1], feats))
			# if args.prob:
			# 	z_std = torch.exp(z_logsigma)
			# 	loss = loss / z_std  #+ z_std
			return loss.detach().numpy(), z_all.detach().numpy() # because we have unnecessary third dimension
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
			return loss.detach().numpy(), y_pred.detach().numpy()

if __name__ == '__main__':
	print(args, '\n')

	# define path for results, checkpoints & plots & create directories
	folder = f'{args.model}_{args.dataset}/n_window{args.n_window}_steps{args.step_size}_eps{args.epochs}'
	# folder = f'studies2.3/{args.model}_{args.dataset}/detectionlvl_{args.q}'
	plot_path = f'{folder}/plots'
	res_path = f'{folder}/results'
	if args.checkpoint is None:
		checkpoints_path = f'{folder}/checkpoints'
	else:
		checkpoints_path = args.checkpoint
	os.makedirs(plot_path, exist_ok=True)
	os.makedirs(res_path, exist_ok=True)

	train_loader, test_loader, labels, ts_lengths = load_dataset(args.dataset)
	feats = train_loader.dataset.shape[1]
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, feats, checkpoints_path)

	# Calculate and print the number of parameters
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'total params: {total_params}, trainable params: {trainable_params}')

	# save arguments and additional info in config file
	with open(f'{folder}/config.txt', 'w') as f:
		f.write(f'{args.model} on {args.dataset}\n \n')
		f.write(str(args)+'\n')
		f.write(f'total params: {total_params}, trainable params: {trainable_params}\n')
		f.write(f'feats: {feats}, \nts_lengths: {ts_lengths}\n')
		f.write(f'optimizer: {optimizer}, \nscheduler: {scheduler}\n')

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainD, testD = trainD.to(torch.float64), testD.to(torch.float64)  # necessary because model in double precision, data should be as well
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'iTransformer'] or 'TranAD' in model.name: 
		trainD = convert_to_windows_train(trainO, model, args.step_size, ts_lengths) 				 # use windows shifted by step size for training
		train_test, _ = convert_to_windows_test(trainO, model, labels=None, w_size=args.n_window)	 # use non-overlapping windows for testing, need this for POT
		testD, labels  = convert_to_windows_test(testD, model, labels, w_size=args.n_window) 		 # use non-overlapping windows for testing

	# if model.name == 'iTransformer':
	# 	summary(model, input_data=trainD, depth=5, verbose=1)
	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = args.epochs; e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, feats, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(checkpoints_path, model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, plot_path)

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, feats, optimizer, scheduler, training=False)

	### Plot curves
	# if not args.test:
	if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
	if feats <= 40:
		plotter(plot_path, testO, y_pred, loss, labels)

	### Scores
	lossT, _ = backprop(0, model, train_test, feats, optimizer, scheduler, training=False)  # need anomaly scores on training data for POT

	# # if step_size > 1, define truth labels per window instead of per time stamp, can also just use non-overlapping windows for testing
	# if args.step_size > 1:
	# 	print(labels.shape, labels[labels[:,0]==1].shape, labels[labels[:,0]==0].shape)
	# 	nb_windows = (len(labels) - args.n_window) // args.step_size
	# 	labels = np.array([(np.sum(labels[i*args.step_size:i*args.step_size+args.n_window], axis=0) >= 1) + 0 for i in range(nb_windows)])
	# 	print(labels.shape, labels[labels[:,0]==1].shape, labels[labels[:,0]==0].shape)

	# local anomaly labels
	df_res_local = pd.DataFrame()
	preds = []
	for i in range(feats):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]  	
		result_local, pred = pot_eval(lt, l, ls, plot_path, f'dim{i}', q=args.q)
		preds.append(pred)
		df_res = pd.DataFrame.from_dict(result_local, orient='index').T
		df_res_local = pd.concat([df_res_local, df_res], ignore_index=True)
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	true_labels = (np.sum(labels, axis=1) >= 1) + 0
	preds = np.array(preds).T
	preds = preds.astype(int)
	labelspred = (np.sum(preds, axis=1) >= 1) + 0
	# plot_ascore(plot_path, 'ascore_local', ascore=loss, labels=true_labels)
	plot_labels(plot_path, 'labels_local', y_pred=labelspred, y_true=true_labels)
	# plot_metrics(plot_path, 'metrics_local', y_pred=labelspred, y_true=true_labels)
	result_local = calc_point2point(predict=labelspred, actual=true_labels)
	result_local1 = {'f1': result_local[0], 'precision': result_local[1], 'recall': result_local[2], 
				  'TP': result_local[3], 'TN': result_local[4], 'FP': result_local[5], 'FN': result_local[6], 
				  'ROC/AUC': result_local[7], 'MCC': result_local[8]}
	result_local1.update({'detection level q': args.q})
	print('local results')
	pprint(result_local1)

	# do majority voting over dimensions for local results instead of inclusive OR
	majority = math.ceil(labels.shape[1] / 2)
	labelspred_maj = (np.sum(preds, axis=1) >= majority) + 0
	plot_labels(plot_path, 'labels_local_maj', y_pred=labelspred_maj, y_true=true_labels)
	# plot_metrics(plot_path, 'metrics_local_maj', y_pred=labelspred_maj, y_true=true_labels)
	result_local = calc_point2point(predict=labelspred_maj, actual=true_labels)
	result_local2 = {'f1': result_local[0], 'precision': result_local[1], 'recall': result_local[2], 
				  'TP': result_local[3], 'TN': result_local[4], 'FP': result_local[5], 'FN': result_local[6], 
				  'ROC/AUC': result_local[7], 'MCC': result_local[8]}
	result_local2.update({'detection level q': args.q})
	print('\nlocal results with majority voting')
	pprint(result_local2)
	temp = np.where(labelspred_maj != true_labels)
	# print(temp, np.all(labelspred_maj == true_labels))

	# global anomaly labels
	result_global, pred2 = pot_eval(lossTfinal, lossFinal, true_labels, plot_path, f'all_dim', q=args.q)
	labelspred_glob = (pred2 >= 1) + 0
	# plot_ascore(plot_path, 'ascore_global', ascore=lossFinal, labels=true_labels)
	plot_labels(plot_path, 'labels_global', y_pred=labelspred_glob, y_true=true_labels)
	# plot_metrics(plot_path, 'metrics_global', y_pred=labelspred_glob, y_true=true_labels)
	metrics_global = calc_point2point(predict=labelspred_glob, actual=true_labels)
	result_global.update(hit_att(loss, labels))
	result_global.update(ndcg(loss, labels))
	result_global.update({'detection level q': args.q})
	print('\nglobal results') 
	pprint(result_global)

	plot_metrics(plot_path, ['local (incl. OR)', 'local (maj. voting)', 'global'], 
			  y_pred=[labelspred, labelspred_maj, labelspred_glob], y_true=true_labels)

	# compare local & global anomaly labels
	compare_labels(plot_path, pred_labels=[labelspred, labelspred_maj], true_labels=true_labels, 
				plot_labels=['Local anomaly\n(inclusive OR)', 'Local anomaly\n(majority voting)'], name='_loc_vs_maj')
	compare_labels(plot_path, pred_labels=[labelspred, labelspred_maj, labelspred_glob], true_labels=true_labels, 
				plot_labels=['Local anomaly\n(inclusive OR)', 'Local anomaly\n(majority voting)', 'Global anomaly'], name='_all')

	# saving results
	df_res_global = pd.DataFrame.from_dict(result_global, orient='index').T
	df_res_global.index = ['global']
	result_local1 = pd.DataFrame.from_dict(result_local1, orient='index').T
	result_local2 = pd.DataFrame.from_dict(result_local2, orient='index').T
	result_local1.index = ['local_all']
	result_local2.index = ['local_all_maj']
	df_res_local = pd.concat([df_res_local, result_local1, result_local2])
	df_res = pd.concat([df_res_local, df_res_global]) 
	df_labels = pd.DataFrame( {'local': labelspred, 'local_maj': labelspred_maj, 'global': labelspred_glob} )

	df_res.to_csv(f'{res_path}/res.csv')	
	df_labels.to_csv(f'{res_path}/pred_labels.csv', index=False)

