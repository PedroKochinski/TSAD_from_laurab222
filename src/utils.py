import os
import torch
import torch.nn as nn

import src.models

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cut_array(percentage, arr):
	print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
	mid = round(arr.shape[0] / 2)
	window = round(arr.shape[0] * percentage * 0.5)
	return arr[mid - window : mid + window, :]

def save_model(folder, model, optimizer, scheduler, epoch, accuracy_list, name=''):
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model{name}.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dataset, dims, window_size, step_size=None, d_model=None, 
			   test=False, path=None, loss='MSE', prob=False, weighted=False):
	""" Load or create a model with the specified parameters.
		Parameters:
		modelname (str): The name of the model class to be loaded or created.
		dims (int): The dimensions of the input data, corresponds to nb of features.
		window_size (int): The window size for the model.
		step_size (int, optional): The step size for the model. Default is None.
		d_model (int, optional): The dimension of the model. Default is None.
		test (bool, optional): Whether to test the model. Default is False.
		path (str, optional): The path to load the model from. Default is None.
		loss (str, optional): The loss function to be used. Default is 'MSE'.
		prob (bool, optional): Whether to use probabilistic modeling. Default is False.
		weighted (bool, optional): Whether to use weighted loss. Default is False.
		Returns:
		tuple: A tuple containing the model, optimizer, scheduler, epoch, and accuracy_list.
		If a pre-trained model exists at the specified path and retraining is not required, 
		the model, optimizer, and scheduler states are loaded from the checkpoint. 
		Otherwise, a new model is created and initialized.
	"""

	model_class = getattr(src.models, modelname)
	if modelname == 'iTransformer':
		model = model_class(dims, window_size, step_size, d_model, loss, prob, weighted).double()
	else:
		model = model_class(dims, window_size, prob).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	if path is not None:
		fname = os.path.join(path, 'model_final.ckpt')
	else:
		fname = f'{modelname}_{dataset}/window_size{window_size}/checkpoints/model_final.ckpt'
	if os.path.exists(fname) and test:
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC} from {fname}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

class EarlyStopper:
	""" Early stopping class to stop training when the validation loss does not improve.
		Parameters:
		patience (int, optional): The number of epochs to wait before stopping. Default is 2.
		min_delta (float, optional): The minimum change in validation loss to be considered an improvement. Default is 0.
	"""
	def __init__(self, patience=2, min_delta=0):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.min_validation_loss = None
	
	def early_stop(self, validation_loss):
		if self.min_validation_loss is None:
			self.min_validation_loss = validation_loss
		elif validation_loss <= (self.min_validation_loss + self.min_delta):
			if validation_loss > self.min_validation_loss:
				self.min_validation_loss = validation_loss
			self.counter += 1
			if self.counter >= self.patience:
				return True
		else:
			self.min_validation_loss = validation_loss
			self.counter = 0
		return False
	

def loss_quantile(output, target, quantile):   
	"""pinball loss function, metric to asses accuracy of quantile predictions"""
	z = target - output
	loss = torch.where(z < 0, quantile * z, (quantile - 1) * z)
	return loss

class combined_loss(nn.Module):
	""" Combined loss function for the iTransformer model.
		Parameters:
		window_size (int): The window size for the model.
		penalty (bool, optional): Whether to include a variability penalty term in the loss function. Default is False.

		The combined loss function consists of three components: Huber loss, quantile loss at 0.25, and quantile loss at 0.75.
		The penalty term is included if the penalty parameter is set to True.
	"""
	def __init__(self, window_size, penalty=False):
		super(combined_loss, self).__init__()
		self.window_size = window_size
		self.penalty = penalty
	
	def forward(self, output, target):
		output1, output2, output3 = torch.split(output, self.window_size, dim=1)
		loss_Huber = torch.nn.HuberLoss(reduction='none')
		huber = loss_Huber(output1, target)
		quantile1 = loss_quantile(output2, target, quantile=0.25)
		quantile2 = loss_quantile(output3, target, quantile=0.75)
		penalty = torch.log( torch.abs(output3 - output2) + 1e-4 )
		if self.penalty:
			loss_tot = huber + quantile1 + quantile2 - 0.01*penalty
		else:
			loss_tot = huber + quantile1 + quantile2
		# print('huber:', huber.mean().item(), 'quantile1:', quantile1.mean().item(), 'quantile2:', quantile2.mean().item(), 'penalty:', penalty.mean().item())
		# print('loss:', loss_tot.mean().item())
		return loss_tot