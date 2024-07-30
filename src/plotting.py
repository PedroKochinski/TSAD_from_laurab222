import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np
from sklearn.metrics import *
import scienceplots

matplotlib.use('Agg')
plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2
plt.rcParams['lines.markersize'] = 3

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

def plot_labels(path, name, y_pred, y_true):
	os.makedirs(path, exist_ok=True)
	fig, ax1 = plt.subplots(figsize=(6,2))
	ax1.plot(smooth(y_pred), 'g.', label='Predicted anomaly')
	ax1.plot(y_true, '--', linewidth=0.3, alpha=0.2)
	ax1.fill_between(np.arange(y_true.shape[0]), y_true, color='blue', alpha=0.3, label='True anomaly')
	plt.legend()
	ax1.set_xlabel('Timestamp')
	ax1.set_ylabel('Labels')
	plt.savefig(f'{path}/{name}.png', dpi=300)
	plt.close()

def plot_ascore(path, name, ascore, labels):
	os.makedirs(path, exist_ok=True)
	
	if ascore.ndim != 1:
		pdf = PdfPages(f'{path}/{name}.pdf')
		for dim in range(ascore.shape[1]):	
			fig, axs = plt.subplots(1, 1, figsize=(6,2))
			axs.plot(smooth(ascore[:,dim]), 'g.')
			ax3 = axs.twinx()
			ax3.plot(labels, '--', linewidth=0.3, alpha=0.5)
			ax3.fill_between(np.arange(labels.shape[0]), labels, color='blue', alpha=0.3, label='True anomaly')
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
	ax3.plot(labels, '--', linewidth=0.3, alpha=0.5)
	ax3.fill_between(np.arange(labels.shape[0]), labels, color='blue', alpha=0.2, label='True anomaly')
	ax1.legend(ncol=1, bbox_to_anchor=(0.7, 1.0))
	ax3.legend(ncol=1, bbox_to_anchor=(0.7, 0.9))
	ax1.set_xlabel('Timestamp')
	ax1.set_ylabel('Anomalies')
	plt.savefig(f'{path}/{name}_averaged.png', dpi=300)
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

def compare_labels(path, labels_loc, labels_glob, labels):
	os.makedirs(path, exist_ok=True)
	plt.plot(labels_loc, '.', label='Local anomaly')
	plt.plot(labels_glob, 'x', label='Global anomaly')
	plt.plot(labels, '--', linewidth=0.3, alpha=0.2)
	plt.fill_between(np.arange(labels.shape[0]), labels, color='blue', alpha=0.3, label='True anomaly')
	plt.xlabel('Timestamp')
	plt.ylabel('Label')
	plt.title('Comparison of predicted anomaly labels')
	plt.legend()
	plt.savefig(f'{path}/compare_labels.png', dpi=300)
	plt.close()

	plt.plot(labels_loc, '--', linewidth=0.5, label='Local anomaly')
	plt.plot(labels_glob, '--', linewidth=0.5, label='Global anomaly')
	plt.plot(labels, linewidth=0.3, alpha=0.2)
	plt.fill_between(np.arange(labels.shape[0]), labels, color='blue', alpha=0.2, label='True anomaly')
	plt.xlabel('Timestamp')
	plt.ylabel('Label')
	plt.title('Comparison of predicted anomaly labels')
	plt.legend(ncol=1, bbox_to_anchor=(0.75, 1.0))
	plt.savefig(f'{path}/compare_labels2.png', dpi=300)
	plt.close()
        