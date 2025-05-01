import matplotlib.pyplot as plt
import os
import torch
from src.constants import *

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

def load_model(modelname, dims, window_size, step_size=None, path=None, prob=False, weighted=False, forecasting=False):
	import src.models
	model_class = getattr(src.models, modelname)
	if modelname == 'iTransformer':
		model = model_class(dims, window_size, step_size, prob, weighted, forecasting).double()
	else:
		model = model_class(dims, window_size, prob).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	if path is not None:
		fname = os.path.join(path, 'model_final.ckpt')
	else:
		fname = f'{args.model}_{args.dataset}/window{args.window}/checkpoints/model_final.ckpt'
	if (os.path.exists(fname) and not args.retrain) or args.test:
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
		# if validation_loss < self.min_validation_loss:
		#     self.min_validation_loss = validation_loss
		#     self.counter = 0
		# elif validation_loss > (self.min_validation_loss + self.min_delta):
		#     self.counter += 1
		#     if self.counter >= self.patience:
		#         return True
		return False