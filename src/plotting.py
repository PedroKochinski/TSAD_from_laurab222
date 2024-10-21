import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import os, torch
import numpy as np
from sklearn.metrics import *
import mplhep as hep

matplotlib.use('Agg')
# plt.style.use(['science', 'ieee'])
# plt.rcParams["text.usetex"] = False
# plt.rcParams['figure.figsize'] = 6, 2
plt.style.use(hep.style.ROOT)
plt.style.use(hep.style.firamath)
# plt.style.use(hep.style.ATLAS)
# plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
plt.rcParams['lines.markersize'] = 4
plt.rcParams['lines.linewidth'] = 2


def plot_accuracies(accuracy_list, folder):
	os.makedirs(folder, exist_ok=True)
	trainAcc = [i[0] for i in accuracy_list]
	lrs = [i[1] for i in accuracy_list]
	plt.xticks(range(len(trainAcc)))
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss', labelpad=10)
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=2, linestyle='-', marker='.')
	plt.legend(loc='lower left')
	ax2 = plt.twinx()
	ax2.plot(range(len(lrs)), lrs, 'r--', label='Learning Rate', linewidth=2, marker='.')
	ax2.yaxis.set_label_position("right")
	ax2.set_ylabel('Learning Rate', rotation=270, ha='right', rotation_mode='default', labelpad=20)
	ax2.legend(loc='upper right')
	plt.tight_layout()
	plt.savefig(f'{folder}/training-graph.pdf')
	plt.clf()

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(path, y_true, y_pred, ascore, labels):
	os.makedirs(path, exist_ok=True)
	if 'TranAD' in path: y_true = torch.roll(y_true, 1, 0)
	pdf = PdfPages(f'{path}/output.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), label='True')
		ax1.plot(smooth(y_p), '-', label='Predicted')
		ax1.set_xlim([0, len(y_p)])
		ax3 = ax1.twinx()
		ax3.plot(l, 'r', alpha=0.2)
		ax3.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.2, label='Anomaly')
		if dim == 0: 
			ax1.legend(ncol=2, bbox_to_anchor=(0.63, 0.93))
			ax3.legend(ncol=1, bbox_to_anchor=(0.05, 0.93))
		ax2.plot(smooth(a_s))
		# ax2.set_ylim(0, 1.0)
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		plt.tight_layout()
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plot_labels(path, name, y_pred, y_true):
	os.makedirs(path, exist_ok=True)
	fig = plt.figure(figsize=(12, 4))
	plt.plot(smooth(y_pred), 'o', label='Predicted anomaly')
	plt.plot(y_true, 'r', alpha=0.2)
	plt.fill_between(np.arange(y_true.shape[0]), y_true, color='r', alpha=0.1, label='True anomaly')
	plt.yticks([0, 1])
	plt.ylim(-0.1, 1.1)
	plt.xlabel('Timestamp')
	plt.ylabel('Labels')
	plt.legend(facecolor='white')
	# plt.legend(ncol=1, bbox_to_anchor=(1.0, 1.0))
	plt.tight_layout()
	plt.savefig(f'{path}/{name}.png', dpi=100)
	plt.close()

def plot_ascore(path, name, ascore, labels):
	os.makedirs(path, exist_ok=True)
	
	if ascore.ndim != 1:
		pdf = PdfPages(f'{path}/{name}.pdf')
		for dim in range(ascore.shape[1]):	
			fig, axs = plt.subplots(1, 1)
			axs.plot(smooth(ascore[:,dim]), 'g.')
			ax3 = axs.twinx()
			ax3.plot(labels, 'r', alpha=0.2)
			ax3.fill_between(np.arange(labels.shape[0]), labels, color='r', alpha=0.1, label='True anomaly')
			axs.set_xlabel('Timestamp')
			axs.set_ylabel('Anomaly score')
			axs.set_title(f'Dimension = {dim}')
			plt.legend(loc='center right')
			plt.tight_layout()
			pdf.savefig(fig)
			plt.close()
		pdf.close()

	fig, ax1 = plt.subplots()
	if ascore.ndim != 1:
		ascore = np.mean(ascore, axis=1)
	ax1.plot(smooth(ascore), 'g.', label='Anomaly score')
	ax3 = ax1.twinx()
	ax3.plot(labels, 'r', alpha=0.2)
	ax3.fill_between(np.arange(labels.shape[0]), labels, color='red', alpha=0.1, label='True anomaly')
	ax1.legend(ncol=1, bbox_to_anchor=(0.7, 1.0))
	ax3.legend(ncol=1, bbox_to_anchor=(0.7, 0.9))
	ax1.set_xlabel('Timestamp')
	ax1.set_ylabel('Anomalies')
	plt.savefig(f'{path}/{name}_averaged.png', dpi=100)
	plt.close()

def plot_metrics(path, name, y_true, y_pred):
	os.makedirs(path, exist_ok=True)
	linestyles = ['--', '-.', ':']
	# confusion matrix
	for i, pred in enumerate(y_pred):
		cm = ConfusionMatrixDisplay.from_predictions(y_true, pred)
		cm.plot(cmap='plasma')
		plt.savefig(f'{path}/cm_{name[i]}.png', dpi=300)
		plt.close()
		plt.clf()
	# ROC curve
	for i, pred in enumerate(y_pred):
		roc_auc = roc_auc_score(y_true, pred)
		fpr, tpr, _ = roc_curve(y_true,  pred)
		plt.plot(fpr, tpr, linestyle=linestyles[i%len(linestyles)], label=f'{name[i]}: \nauc = {roc_auc:.3f}')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	plt.legend()
	plt.savefig(f'{path}/roc.png', dpi=100)
	plt.close()
	plt.clf()
	# precision-recall curve
	for i, pred in enumerate(y_pred):
		ap = average_precision_score(y_true, pred)
		precision, recall, _ = precision_recall_curve(y_true, pred)
		pr_auc = auc(recall, precision)
		plt.plot(recall, precision, linestyle=linestyles[i%len(linestyles)], label=f'{name[i]}: \naverage precision = {ap:.3f} \nauc = {pr_auc:.3f}')	
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-recall curve')
	plt.legend()
	plt.savefig(f'{path}/pr.png', dpi=100)
	plt.close()
	plt.clf()

def compare_labels(path, pred_labels, true_labels, plot_labels, name=''):
	os.makedirs(path, exist_ok=True)
	fig = plt.figure(figsize=(12, 4))
	markers = ['o', 'x', '*']
	for i, (data, lab) in enumerate(zip(pred_labels, plot_labels)):
		plt.plot(data, marker=markers[i%len(markers)], linestyle='None', label=lab)
	plt.plot(true_labels, 'r', alpha=0.2)
	plt.fill_between(np.arange(true_labels.shape[0]), true_labels, color='r', alpha=0.1, label='True anomaly')
	plt.yticks([0, 1])
	plt.ylim(-0.1, 1.1)
	plt.xlabel('Timestamp')
	plt.ylabel('Label')
	plt.title('Comparison of predicted anomaly labels')
	# plt.legend(loc='center right', facecolor='white')
	plt.legend(ncol=1, bbox_to_anchor=(1.0, 1.0))
	plt.tight_layout()
	plt.savefig(f'{path}/compare_labels{name}.png', dpi=100)
	plt.close()

	fig = plt.figure(figsize=(13, 4))
	linestyles = ['--', '-.', ':']
	for i, (data, lab) in enumerate(zip(pred_labels, plot_labels)):
		plt.plot(data, linestyle=linestyles[i%len(linestyles)], label=lab)
	plt.plot(true_labels, 'r', alpha=0.2)
	plt.fill_between(np.arange(true_labels.shape[0]), true_labels, color='r', alpha=0.1, label='True anomaly')
	plt.yticks([0, 1])
	plt.ylim(-0.1, 1.1)
	plt.xlabel('Timestamp')
	plt.ylabel('Label')
	plt.title('Comparison of predicted anomaly labels')
	plt.legend(ncol=1, bbox_to_anchor=(1.0, 1.0))
	plt.tight_layout()
	plt.savefig(f'{path}/compare_labels2{name}.png', dpi=100)
	plt.close()
        

