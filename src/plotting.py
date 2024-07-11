import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np
from sklearn.metrics import *
import scienceplots

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

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
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3, label='Anomaly')
		if dim == 0: 
			ax1.legend(ncol=2, bbox_to_anchor=(0.6, 0.95))
			ax3.legend(ncol=1, bbox_to_anchor=(0.22, 1.38))
		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		# ax2.set_ylim(0, 1.0)
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plot_labels(path, name, pred_labels, true_labels):
	os.makedirs(path, exist_ok=True)
	fig, ax1 = plt.subplots(figsize=(6,2))
	ax1.plot(smooth(pred_labels), linewidth=0.2, color='g', label='Predicted anomaly')
	ax3 = ax1.twinx()
	ax3.plot(true_labels, '--', linewidth=0.3, alpha=0.5)
	ax3.fill_between(np.arange(true_labels.shape[0]), true_labels, color='blue', alpha=0.3, label='True anomaly')
	plt.legend()
	ax1.set_xlabel('Timestamp')
	ax1.set_ylabel('Labels')
	plt.savefig(f'{path}/{name}.png', dpi=300)
	plt.close()

def plot_ascore(path, name, ascore, labels):
	os.makedirs(path, exist_ok=True)
	fig, axs = plt.subplots(ascore.shape[1], 1, figsize=(10,10), sharex=True, sharey=True)
	for dim in range(ascore.shape[1]):
		a_s = ascore[:, dim]
		if ascore.shape[1] == 1:
			axs.plot(smooth(a_s), linewidth=0.2, color='g')
			# axs.set_ylim(0, 1.0)
		else:
			axs[dim].plot(smooth(a_s), linewidth=0.2, color='g')
			# axs[dim].set_ylim(0, 1.0)
	fig.supxlabel('Timestamp')
	fig.supylabel('Anomaly Score')
	plt.tight_layout()
	plt.savefig(f'{path}/{name}.png', dpi=300)
	plt.close()

	fig, ax1 = plt.subplots()
	# ascore is still multidimensional, wanna reduce that but first smooth them
	for dim in range(ascore.shape[1]):
		ascore[:, dim] = smooth(ascore[:, dim])
	ascore = np.mean(ascore, axis=1)
	ax1.plot(ascore, linewidth=0.2, color='g', label='Anomaly score')
	ax3 = ax1.twinx()
	ax3.plot(labels, '--', linewidth=0.3, alpha=0.5)
	ax3.fill_between(np.arange(labels.shape[0]), labels, color='blue', alpha=0.3, label='Anomaly')
	plt.legend()
	ax1.set_xlabel('Timestamp')
	ax1.set_ylabel('Anomalies')
	plt.savefig(f'{path}/{name}_withlabels.png', dpi=300)
	plt.close()

def plot_metrics(path, name, y_true, y_pred):
	os.makedirs(path, exist_ok=True)
	# confusion matrix
	cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
	cm.plot()
	plt.savefig(f'{path}/cm_{name}.png', dpi=300)
	plt.close()
	plt.clf()
	# ROC curve
	roc_auc = roc_auc_score(y_true, y_pred)
	fpr, tpr, _ = roc_curve(y_true,  y_pred)
	plt.plot(fpr,tpr,label=f'auc = {roc_auc:.3f}')	
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	plt.legend()
	plt.savefig(f'{path}/roc_{name}.png', dpi=300)
	plt.close()
	plt.clf()
	# precision-recall curve
	ap = average_precision_score(y_true, y_pred)
	precision, recall, _ = precision_recall_curve(y_true, y_pred)
	pr_auc = auc(recall, precision)
	plt.plot(recall, precision, label=f'average precision = {ap:.3f} \nauc = {pr_auc:.3f}')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-recall curve')
	plt.legend()
	plt.savefig(f'{path}/pr_{name}.png', dpi=300)
	plt.close()
	plt.clf()

