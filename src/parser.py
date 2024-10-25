import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--n_window', 
					metavar='-nw', 
					type=int, 
					required=False,
					default=10,
                    help="number of timestamps in a window")
parser.add_argument('--step_size', 
					metavar='-step', 
					type=int, 
					required=False,
					default=1,
                    help="step size for sliding window")
parser.add_argument('--q', 
					metavar='-q', 
					type=float, 
					required=False,
					default=1e-5,
                    help="detection level (risk) for POT method")
parser.add_argument('--epochs', 
					metavar='-e', 
					type=int, 
					required=False,
					default=5,
                    help="number of training epochs")
parser.add_argument('--checkpoint', 
                    required=False,
                    default=None,
					help="path to pretrained model checkpoint")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--prob', 
					action='store_true', 
					help="model gives back probabilistic output")
parser.add_argument('--f', default=None, help='dummy argument for jupyter notebooks')
args = parser.parse_args()