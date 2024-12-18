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

def save_model(folder, model, optimizer, scheduler, epoch, accuracy_list):
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims, n_window, step_size=None, path=None, prob=False, weighted=False):
	import src.models
	model_class = getattr(src.models, modelname)
	if modelname == 'iTransformer':
		model = model_class(dims, n_window, step_size, prob, weighted).double()
	else:
		model = model_class(dims, n_window, prob).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	if path is not None:
		fname = os.path.join(path, 'model.ckpt')
	else:
		fname = f'{args.model}_{args.dataset}/n_window{args.n_window}/checkpoints/model.ckpt'
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
