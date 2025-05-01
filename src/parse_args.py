import argparse

def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Time Series Anomaly Detection')
	parser = argparse.ArgumentParser(description='Time Series Anomaly Detection')
	
	# Dataset and model arguments
	parser.add_argument('--dataset', type=str, required=False, default='GECCO', help="give name of dataset")
	parser.add_argument('--model', type=str, required=False, default='iTransformer', help="model name")
	parser.add_argument('--checkpoint', type=str, required=False, default=None, help="path to pretrained model checkpoint folder")

	# Training parameters
	parser.add_argument('--window_size', '-w', type=int, required=False, default=10, help="number of timestamps in a window")
	parser.add_argument('--step_size', '-s', type=int, required=False, default=1, help="step size for sliding window")
	parser.add_argument('--d_model', '-d', type=int, required=False, default=2, help="internal model size for iTransformer model")
	parser.add_argument('--epochs', '-e', type=int, required=False, default=5, help="number of training epochs")
	parser.add_argument('--feats', type=int, required=False, default=-1, help="number of features to be used")
	parser.add_argument('--q', type=float, required=False, default=1e-5, help="detection level (risk) for POT method")
	parser.add_argument('--k', type=int, required=False, default=-1, help="fold for cross-validation")
	
	# Boolean flags
	parser.add_argument('--test', action='store_true', help="test the model")
	parser.add_argument('--retrain', action='store_true', help="retrain the model")
	parser.add_argument('--less', action='store_true', help="train using less data")
	parser.add_argument('--forecasting', action='store_true', required=False, help="forecasting AD instead of reconstruction")
	parser.add_argument('--enc', action='store_true', help="use additional time encoder covariate")
	parser.add_argument('--shuffle', action='store_true', help="shuffle the train/valid data")

	# Loss function
	parser.add_argument('--loss', type=str, required=False, default='MSE', 
					 help="choose loss function: either 'MSE', 'Huber', 'softdtw' of 'softdtw_norm")

	# Miscellaneous
	parser.add_argument('--name', type=str, required=False, default=None, help="name of the output folder")
	parser.add_argument('--f', default=None, help='dummy argument for jupyter notebooks')
	
	return parser.parse_args()

