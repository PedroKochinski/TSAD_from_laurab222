#!/bin/bash

# for m in OmniAnomaly DAGMM iTransformer TranAD  #  LSTM_AD MAD_GAN
# do
#     echo "Model $m"
#     python main.py --model $m --window_size 10 --dataset GECCO --step_size 1 --retrain --feats -1
# done

# echo "data set IEEECIS_new2.2"
# for i in {1..5}
# do
#     echo "rep $i"
#     # echo "data set $d"
#     python main.py --model TranAD --window_size 10 --dataset IEEECIS_new2.2 --step_size 1 --epochs 200 --retrain --feats 30 --k $i --name $i
#     # python main.py --model iTransformer --window_size 10 --dataset IEEECIS_new2.2 --step_size 1 --epochs 200 --retrain --feats 30 --k $i --name latent2_$i
#     # python main.py --model iTransformer --window_size 100 --dataset IEEECIS_new2.2 --step_size 50 --epochs 200 --retrain --feats 30 --k $i --name latent2_$i
# done

# for d in SMAP_new MSL_new creditcard creditcard_normal # GECCO GECCO_normal # UCR SWaT_1D creditcard_normal SMAP_new creditcard
# do
#     echo "data set $d"
#     for i in {1..5}
#     do  
#         echo "rep $i"
#         # python main.py --model TranAD --window_size 10 --dataset $d --step_size 1 --epochs 5 --retrain --feats -1 --k $i --name $i
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 1 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss5_$i --less
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 5 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss5_$i --less
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 50 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss5_$i --less
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 10 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss5_$i --less
#     done
# done

# for d in GECCO GECCO_normal
# do
#     echo "data set $d"
#     for i in {1..5}
#     do  
#         echo "rep $i"
#         # python main.py --model TranAD --window_size 10 --dataset $d --step_size 1 --epochs 5 --retrain --feats -1 --k $i --name newloss3_$i
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 1 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss6_$i
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 5 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss6_$i
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 50 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss6_$i
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 10 --epochs 5 --retrain --feats -1 --k $i --name latent2_newloss6_$i
#     done
# done

# for d in SMD # SWaT UCR GECCO ATLAS_TS SMD MSL_new
# do
#     echo "data set $d"
#     for i in {1..5}
#     do  
#         echo "rep $i"
#         # python main.py --model TranAD --window_size 10 --dataset $d --step_size 1 --epochs 200 --test --feats -1 --k $i --name $i --less
#         python main.py --model iTransformer --window_size 10 --dataset $d --step_size 1 --epochs 200 --test --feats -1 --k $i --name latent2_$i --less
#         python main.py --model iTransformer --window_size 100 --dataset $d --step_size 50 --epochs 200 --test --feats -1 --k $i --name latent2_$i --less
#     done
# done

# # for d in  # SMD SMAP_new UCR ATLAS_TS SWaT
# do
#     echo "data set $d"
#     python main.py --model USAD --window_size 10 --dataset $d --step_size 1 --retrain --feats -1 
# done

# for d in SMD SMAP_new UCR ATLAS_TS
# do
#     echo "data set $d"
#     python main.py --model MERLIN --window_size 10 --dataset $d --step_size 1 --retrain --feats -1 
# done

# python main.py --model TranAD --window_size 10 --dataset creditcard_normal --step_size 1 --epochs 5 --retrain --feats -1 --k -1 --name test
# python main.py --model iTransformer --window_size 10 --dataset WADI --step_size 1 --epochs 5 --retrain --feats 30 --k -1 --name latent2_test
# python main.py --model iTransformer --window_size 100 --dataset WADI --step_size 50 --epochs 5 --retrain --feats 30 --k -1 --name latent2_test
# python main.py --model iTransformer --window_size 10 --dataset SWaT --step_size 1 --epochs 5 --retrain --feats 30 --k -1 --name latent2_test
# python main.py --model iTransformer --window_size 100 --dataset SWaT --step_size 50 --epochs 5 --retrain --feats 30 --k -1 --name latent2_test

# python main.py --model iTransformer --window_size 2000 --dataset ATLAS_DQM_TS --retrain --step_size 1000 --feats -1 --name train_all
# python main.py --model iTransformer --window_size 1000 --dataset WADI --retrain --step_size 500 --feats -1 --name latent2_1 --k 1

# # MSL_new
# python main.py --model iTransformer --dataset MSL_new -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_5 --retrain --forecasting 
# python main.py --model iTransformer --dataset MSL_new -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_4 --test --forecasting 
# python main.py --model iTransformer --dataset MSL_new -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_5 --retrain --forecasting 
# W_minus=10 
# python main.py --model iTransformer --dataset MSL_new -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_5 --test --forecasting
# python main.py --model iTransformer --dataset MSL_new -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats --k $kfold --name rep_5 --retrain --forecasting
