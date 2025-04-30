
# README

This repository allows to run and compare multipe time series anomaly detection algorithms and contains implementations
For forecasting-based anomaly detection, switch to branch 'forecasting' please.

## Info
Part of this repository is based on the code corresponding to the paper "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data" (S.Tuli et al., VLDB 2022). The code can be found [here](https://github.com/imperial-qore/TranAD). Thanks a lot!

The iTransformer model used in this code is based on the implementation [here](https://github.com/thuml/iTransformer), based on the paper "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (Y.Liu et al., ICLR 2024).


## Installation
This code needs Python-3.8 or higher.
The provided environment.yml file can be used to set up the environment (using conda or mamba) using:
```bash
conda env create -f environment.yml
```

## Dataset Preprocessing
Preprocess all datasets except for IEEECIS using the command
```bash
python3 preprocess.py creditcard GECCO SMAP MSL SWaT WADI SMD UCR 
```
Distribution rights to some datasets (i.e. SWaT and WADI) are not be available. All other datasets can be found online, following the references in the paper.
Further preprocessing is needed for IEEECIS, which is done in 'check_data.ipynb'.


## Result Reproduction
To run a model on a dataset, run the following command:
```bash
python main.py --model <model> --dataset <dataset> --retrain
```
where `<model>` can be either of 'TranAD', 'GDN', 'MAD_GAN', 'MTAD_GAT', 'MSCRED', 'USAD', 'OmniAnomaly', 'LSTM_AD', and dataset can be one of 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'MSDS', 'MBA', 'UCR' and 'NAB. To train with less data, (maximum of 10k time stamps), use the following command 
```bash
python3 main.py --model <model> --dataset <dataset> --retrain --less
```

When working with forecasting-based anomaly detection, it is necessary to switch to the branch `forecasting` and add the argument `--forecasting` to the command:
```bash
python3 main.py --model <model> --dataset <dataset> --retrain --forecasting
```
The step size is by default always 1 when forecasting time series.

Further arguments, their description and default values can be found  `src/parse_args.py`.

The output will provide training time and anomaly detection performance. For example:
```bash
Arguments:

{'checkpoint': None,
 'd_model': 2,
 'dataset': 'GECCO_normal',
 'enc': False,
 'epochs': 2,
 'f': None,
 'feats': -1,
 'k': -1,
 'less': True,
 'loss': 'MSE',
 'model': 'iTransformer',
 'name': 'test_new',
 'q': 1e-05,
 'retrain': True,
 'shuffle': False,
 'step_size': 1,
 'test': False,
 'window_size': 10}

CUDA available: False
MPS (Apple Silicon GPU) is available: True 

train shape with windows: (9991, 10, 9)
test shape with windows: (1000, 10, 9)
labels shape: (10000, 9)
Creating new model: iTransformer
total params: 3036, trainable params: 3036
Training iTransformer on GECCO_normal
Epoch 0,        L_train = 0.08838528090700046,          L_valid = 0,    LR = 0.0001                                             
Epoch 1,        L_train = 0.08258313718844805,          L_valid = 0,    LR = 0.0001                                             
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:19<00:00,  9.52s/it]
Training time:    19.0772 s or 0.32 min 
Testing iTransformer on GECCO_normal
Local results with 1 anomalous dimensions for anomaly
{'FN': 46,
 'FP': 79,
 'MCC': 0.8678293548295123,
 'ROC/AUC': 0.9478345746650794,
 'TN': 9442,
 'TP': 433,
 'detection level q': 1e-05,
 'f1': 0.8738597709843012,
 'precision': 0.8457031084823612,
 'recall': 0.9039665782052907}
Local results with 5 anomalous dimensions for anomaly
{'FN': 479,
 'FP': 0,
 'MCC': 0.0,
 'ROC/AUC': 0.5,
 'TN': 9521,
 'TP': 0,
 'detection level q': 1e-05,
 'f1': 0.0,
 'precision': 0.0,
 'recall': 0.0}
Global results with 9 anomalous dimensions for anomaly
{'FN': 479,
 'FP': 3,
 'Hit@100%': 1.0,
 'Hit@150%': 1.0,
 'MCC': -0.003885547794772807,
 'NDCG@100%': 1.0,
 'NDCG@150%': 1.0,
 'ROC/AUC': 0.49984245352378953,
 'TN': 9518,
 'TP': 0,
 'detection_level_q': 1e-05,
 'f1': 0.0,
 'precision': 0.0,
 'recall': 0.0,
 'test_loss': 0.08496810780089022,
 'threshold': 3.3204767317791255,
 'train_loss': 0.0804248683839156,
 'train_time': 19.07716202735901}
```

