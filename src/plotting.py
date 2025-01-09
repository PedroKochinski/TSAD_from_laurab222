import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import os, torch
import numpy as np
from sklearn.metrics import *
import mplhep as hep

matplotlib.use('Agg')
# plt.style.use(['science', 'ieee'])
plt.style.use(hep.style.ROOT)
plt.style.use(hep.style.firamath)
# plt.style.use(hep.style.ATLAS)
plt.rcParams['lines.markersize'] = 4
plt.rcParams['lines.linewidth'] = 2

features_dict = {
	'ATLAS_DQM_TS': [r'EMBA $\hat{Q}$, mean', r'EMBA $\hat{Q}$, std', r'EMBA $\hat{\tau}$, mean', r'EMBA $\hat{\tau}$, std', r'EMBC $\hat{Q}$, mean', r'EMBC $\hat{Q}$, std',
					 r'EMBC $\hat{\tau}$, mean',  r'EMBC $\hat{\tau}$, std', r'EMECA $\hat{Q}$, mean', r'EMECA $\hat{Q}$, std', r'EMECA $\hat{\tau}$, mean', r'EMECA $\hat{\tau}$, std',
					 r'EMECC $\hat{Q}$, mean', r'EMECC $\hat{Q}$, std', r'EMECC $\hat{\tau}$, mean', r'EMECC $\hat{\tau}$, std']
}


def add_atlas(ax, lines, status='Internal'):
    """ add_atlas - Adds the atlas label to an axis and follows
    it with any additional lines of text the user wishes to put on the plot.

    Arguments:
    ax (axis object) - Axis object for the plot
    lines (list of strings) - List of lines to place with ATLAS label

    No returns
    """

    # Some hardcoded constants for positioning text in the plot
    left_edge = ax.get_position().x0 - 0.12
    # top = 0.895
    # spacing = 0.057
    top = 2.2 #1.95 # 0.91
    spacing = 0.3

	# Write text
    hep.atlas.text(status, ax=ax, fontsize=20)
    for i, ln in enumerate(lines):
        vertical = top - i * spacing
        ax.text(left_edge, vertical, ln, transform=ax.transAxes, ha='left', va='top', fontsize=17)

def plot_accuracies(accuracy_list, folder):
	os.makedirs(folder, exist_ok=True)
	trainAcc = [i[0] for i in accuracy_list]
	lrs = [i[-1] for i in accuracy_list]
	plt.xticks(range(len(trainAcc)))
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss', ha='center')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=2, linestyle='-', marker='o')
	# plt.legend(loc='lower left')
	plt.legend(bbox_to_anchor=(0.623, 0.1))
	ax2 = plt.twinx()
	ax2.plot(range(len(lrs)), lrs, 'r--', label='Learning Rate', linewidth=2, marker='o')
	ax2.yaxis.set_label_position("right")
	ax2.set_ylabel('Learning Rate', rotation=270, labelpad=25, ha='center', va='top')
	# ax2.legend(loc='upper right')
	ax2.legend(bbox_to_anchor=(0.46, 0.15))
	plt.tight_layout()
	plt.savefig(f'{folder}/training-graph.pdf')
	plt.clf()

def plot_losses(accuracy_list, folder):
	os.makedirs(folder, exist_ok=True)
	lossT = [i[0] for i in accuracy_list]
	lossV = [i[1] for i in accuracy_list]
	epochs = range(1, len(lossT)+1)

	plt.plot(epochs, lossT, label='Average Training Loss', marker='o')
	plt.plot(epochs, lossV, label='Average Validation Loss', marker='o')
	plt.xticks(epochs)
	plt.xlabel('Epochs')
	plt.ylabel('Average Loss')
	plt.ylim(bottom=0)
	# plt.yscale('log')

	plt.legend()
	plt.tight_layout()
	plt.savefig(f'{folder}/losses.pdf')
	plt.clf()

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(path, y_true, y_pred, ascore, labels, ts_length=[], name='output'):
	os.makedirs(path, exist_ok=True)
	# if 'TranAD' in path: y_true = torch.roll(y_true, 1, 0)
	pdf = PdfPages(f'{path}/{name}.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), constrained_layout=True)
		ax1.set_ylabel('Value', labelpad=20, ha='center', va='center')
		ax1.set_title(f'Dimension = {dim}')
		ax1.plot(smooth(y_t), label='True')
		ax1.plot(smooth(y_p), '-', label='Predicted')
		ax1.set_xlim([0, len(y_p)])
		ax3 = ax1.twinx()
		ax3.plot(l, 'r', alpha=0.2)
		ax3.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.2, label='Anomaly')
		if ts_length != [] and len(ts_length) > 1:
			# sum up previous entries in ts_length to get the end of each time series
			start = 0
			for x in np.cumsum(ts_length):
				if start == 0:
					ax1.axvline(x=x, color='k', linestyle=':', label='End of time series')
				else:
					ax1.axvline(x=x+start, color='k', linestyle=':')
				start += x
		if dim == 0: 
			if ts_length != [] and len(ts_length) > 1:
				ax1.legend(ncol=2, bbox_to_anchor=(0.57, 1.55), loc='upper right',  borderaxespad=0., frameon=False)
				ax3.legend(ncol=1, bbox_to_anchor=(0.95, 1.4), loc='upper right',  borderaxespad=0., frameon=False)
			else:
				ax1.legend(ncol=2, bbox_to_anchor=(0.35, 1.25), loc='upper right',  borderaxespad=0., frameon=False)
				ax3.legend(ncol=1, bbox_to_anchor=(0.95, 1.25), loc='upper right', borderaxespad=0., frameon=False)
		ax2.plot(smooth(a_s), 'g-')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score', labelpad=20, ha='center', va='center')
		fig.align_ylabels([ax1, ax2])  # Align y-labels
		# plt.tight_layout()
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plotter2(path, x_true, x_pred, ascore, dataset, y_pred=None, y=None, name=''):
	os.makedirs(path, exist_ok=True)
	if dataset in features_dict.keys():
		features = features_dict[dataset]
	else:
		features = [f'Dim {i}' for i in range(x_true.shape[1])]
	dims = len(features) + 1  # because we plot ascore in the second last dimension
	if y is not None and y_pred is not None:
		dims += 2
	size = int(dims * 1.3)

	fig, axs = plt.subplots(dims, 1, figsize=(17, size), sharex=True, constrained_layout=True)
	if 'ATLAS' in dataset:
		add_atlas(axs[0], ['Data October 2023, 'r'$\sqrt{s_{NN}}= 5.36$'' TeV', 'HardProbes stream'])
	for dim, feat in enumerate(features):  # iterate through the features we're using
		axs[dim].plot(x_true[:, dim], label='True')
		axs[dim].plot(x_pred[:, dim], '--', label='Predicted')
		axs[dim].set_ylabel(feat, rotation=0, ha='right', rotation_mode='default', labelpad=5)
		axs[dim].yaxis.set_label_coords(-0.1, 0.5)
		if 'ATLAS' in dataset:
			axs[dim].set_ylim(-1, 1)
	axs[0].legend(ncol=2, bbox_to_anchor=(0.8, 1.02), loc='lower center', borderaxespad=0., frameon=False)

	if y is not None and y_pred is not None:  # plot the target variable in last dimension if we have truth labels
		axs[-3].plot(ascore, '-', color='tab:green')
		axs[-3].set_ylabel('Anomaly score', rotation=0, ha='right', rotation_mode='default', labelpad=5)
		axs[-2].plot(y_pred, '-', color='tab:green', alpha=0.7)
		axs[-2].set_ylabel('Predicted anomalies', rotation=0, ha='right', rotation_mode='default', labelpad=5)
		axs[-1].plot(y, '-', color='tab:red')
		axs[-1].set_ylabel('True anomalies', rotation=0, ha='right', rotation_mode='default', labelpad=5)
		axs[-3].yaxis.set_label_coords(-0.1, 0.5)
		axs[-2].yaxis.set_label_coords(-0.1, 0.5)
		axs[-1].yaxis.set_label_coords(-0.1, 0.5)
	else:
		axs[-1].plot(ascore, '-', color='tab:green')
		axs[-1].set_ylabel('Anomaly score', rotation=0, ha='right', rotation_mode='default', labelpad=5)
		axs[-1].yaxis.set_label_coords(-0.1, 0.5)

	if 'ATLAS' in dataset:
		axs[-1].set_xlabel('Events')
	elif 'IEEECIS' in dataset:
		axs[-1].set_xlabel('Transactions')
	else:
		axs[-1].set_xlabel('Timestamp')
	fig.savefig(f'{path}/output2{name}.png', dpi=300, facecolor='white')
	plt.close()

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
		plt.plot(fpr, tpr, linestyle=linestyles[i%len(linestyles)], 
		   label=f'{name[i]}: \nauc = {roc_auc:.3f}', linewidth=2)
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
		plt.plot(recall, precision, linestyle=linestyles[i%len(linestyles)], 
		   label=f'{name[i]}: \naverage precision = {ap:.3f} \nauc = {pr_auc:.3f}', linewidth=2)	
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-recall curve')
	plt.legend()
	plt.savefig(f'{path}/pr.png', dpi=100)
	plt.close()
	plt.clf()

def compare_labels(path, pred_labels, true_labels, plot_labels, name=''):
	os.makedirs(path, exist_ok=True)
	fig = plt.figure(figsize=(13, 4))
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
        

