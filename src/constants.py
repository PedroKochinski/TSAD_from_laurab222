from src.parse_args import parse_arguments
from src.folderconstants import *

args = parse_arguments()
# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1)],
        'SWaT_1D': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'SMAP_new': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'MSL_new': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
		'MSDS': [(0.91, 1), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
        'IEEECIS': [(0.95, 1), (0.99, 1)],
        'IEEECIS_new': [(0.96, 1), (0.99, 1)],
        'IEEECIS_new1.2': [(0.96, 1), (0.99, 1)],
        'IEEECIS_new1.3': [(0.96, 1), (0.99, 1)],
        'IEEECIS_new2': [(0.96, 1), (0.99, 1)],
        'IEEECIS_new2.2': [(0.96, 1), (0.99, 1)],
        'IEEECIS_new2.3': [(0.96, 1), (0.99, 1)],
        'IEEECIS_pca': [(0.96, 1), (0.99, 1)],
        'IEEECIS_pca_scaled': [(0.96, 1), (0.99, 1)],
        'ATLAS_TS':  [(0.9995, 1.04), (0.99995, 1.06)],
        'ATLAS_DQM_TS':  [(0.9995, 1.04), (0.99995, 1.06)],
        'GECCO':  [(0.9995, 1.04), (0.99995, 1.06)],
        'lorenzetti':  [(0.9995, 1.04), (0.99995, 1.06)],
        'creditcard': [(0.9995, 1.04), (0.99995, 1.06)],
        'creditcard_time': [(0.9995, 1.04), (0.99995, 1.06)],
	}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'synthetic': 0.0001, 
		'SWaT': 0.008, 
        'SWaT_1D': 0.008,
		'SMAP': 0.001, 
        'SMAP_new': 0.001, 
		'MSL': 0.002, 
		'MSL_new': 0.002, 
		'WADI': 0.0001, 
		'MSDS': 0.001, 
		'UCR': 0.006, 
		'NAB': 0.009, 
		'MBA': 0.001, 
        'IEEECIS': 0.001,
        'IEEECIS_new': 0.001,
        'IEEECIS_new1.2': 0.001,
        'IEEECIS_new1.3': 0.001,
        'IEEECIS_new2': 0.001,
        'IEEECIS_new2.2': 0.001,
        'IEEECIS_new2.3': 0.001,
        'IEEECIS_pca': 0.001,
        'IEEECIS_pca_scaled': 0.001,
		'ATLAS_TS': 0.0001,
        'ATLAS_DQM_TS': 0.0001,
        'GECCO': 0.0001,
        'lorenzetti': 0.0001,
        'creditcard': 0.0001,
        'creditcard_time': 0.0001,
	}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMD': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
        'SWaT_1D': (95, 10),
		'SMAP': (97, 5000),
		'SMAP_new': (97, 5000),
		'MSL': (97, 150),
        'MSL_new': (97, 150),
		'WADI': (99, 1200),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
        'IEEECIS': (98, 2),
        'IEEECIS_new': (97, 3),
        'IEEECIS_new1.2': (97, 3),
        'IEEECIS_new1.3': (97, 3),
        'IEEECIS_new2': (97, 3),
        'IEEECIS_new2.2': (97, 3),
        'IEEECIS_new2.3': (97, 3),
        'IEEECIS_pca': (97, 3),
        'IEEECIS_pca_scaled': (97, 3),
        'ATLAS_TS': (98, 2000),
        'ATLAS_DQM_TS': (98, 2000),
		'GECCO': (98, 2000),
        'lorenzetti': (98, 2000),
        'creditcard': (99, 2000),
        'creditcard_time': (99, 2000),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
debug = 9