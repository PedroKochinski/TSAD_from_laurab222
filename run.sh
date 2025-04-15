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


# kfold=-1
# epochs=10
# feats=-1


# # MSL_new
# python main.py --model iTransformer --dataset MSL_new -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_5 --retrain --forecasting 
# python main.py --model iTransformer --dataset MSL_new -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_4 --test --forecasting 
# python main.py --model iTransformer --dataset MSL_new -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_5 --retrain --forecasting 
# W_minus=10 
# python main.py --model iTransformer --dataset MSL_new -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_5 --test --forecasting
# python main.py --model iTransformer --dataset MSL_new -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats --k $kfold --name rep_5 --retrain --forecasting
# SMAP_new
# python main.py --model iTransformer --dataset SMAP_new -w 15 -s 8 -d 15 -e $epochs --feats $feats --k 4 --name rep_4 --retrain


# ,dataset,model,max_MCC,max_MCC_mode,window,steps,dmodel,max_MCC_std
# 1,creditcard,window10_steps1_dmodel2_feats-1_eps10_MSE,0.237,local_all,10,1,2,0.237 $\pm$ 0.016
# 0,creditcard_normal,window10_steps1_dmodel2_feats-1_eps10,0.241,local_all,10,1,2,0.241 $\pm$ 0.019
# 1,GECCO,window12_steps1_dmodel2_feats-1_eps10_MSE,0.662,local_all,12,1,2,0.662 $\pm$ 0.03
# 1,GECCO_normal,window12_steps1_dmodel2_feats-1_eps10,0.760,local_all,12,1,2,0.76 $\pm$ 0.004
# 2,IEEECIS_new2.2,window10_steps1_dmodel10_feats30_eps10_MSE,0.629,local_all,10,1,10,0.629 $\pm$ 0.032
# 3,MSL_new,window96_steps10_dmodel96_feats-1_eps10,0.935,local_all_maj,96,10,96,0.935 $\pm$ 0.011
# 4,SMAP_new,window15_steps2_dmodel15_feats-1_eps10,0.826,local_all_maj,15,2,15,0.826 $\pm$ 0.091
# 5,SMD,window350_steps35_dmodel70_feats-1_eps10,0.967,local_all,350,35,70,0.967 $\pm$ 0.01
# 6,SWaT,window51_steps5_dmodel10_feats-1_eps10,0.964,local_all_maj,51,5,10,0.964 $\pm$ 0.004
# 7,SWaT_1D,window10_steps5_dmodel2_feats-1_eps10,0.805,-,10,5,2,0.805 $\pm$ 0.006
# 8,UCR,window10_steps5_dmodel2_feats-1_eps10,0.775,-,10,5,2,0.775 $\pm$ 0.01
# 9,WADI,window44_steps4_dmodel44_feats30_eps10,0.825,global,44,4,44,0.825 $\pm$ 0.024

epochs=10

for i in {3..5}
do
    python main.py --model iTransformer --dataset GECCO_normal -w 12 -s 1 -d 2 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain > iTransformer_new/out_GECCO_normal_$i.log
    python main.py --model iTransformer --dataset IEEECIS_new2.2 -w 10 -s 1 -d 10 -e $epochs --feats 30 --k $i --loss softdtw --name rep_$i --retrain > iTransformer_new/out_IEEECIS_$i.log
    python main.py --model iTransformer --dataset MSL_new -w 96 -s 10 -d 96 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain --less > iTransformer_new/out_MSL_$i.log
    python main.py --model iTransformer --dataset SMAP_new -w 15 -s 2 -d 15 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain > iTransformer_new/out_SMAP_$i.log
    python main.py --model iTransformer --dataset SMD -w 350 -s 35 -d 70 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain --less > iTransformer_new/out_SMD_$i.log
    python main.py --model iTransformer --dataset SWaT_small -w 51 -s 5 -d 10 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain > iTransformer_new/out_SWaT_small_$i.log
    python main.py --model iTransformer --dataset SWaT_1D -w 10 -s 5 -d 2 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain > iTransformer_new/out_SWaT_1D_$i.log
    python main.py --model iTransformer --dataset UCR -w 10 -s 5 -d 2 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain > iTransformer_new/out_UCR_$i.log
    python main.py --model iTransformer --dataset WADI -w 44 -s 4 -d 44 -e $epochs --feats 30 --k $i --loss softdtw --name rep_$i --retrain --less > iTransformer_new/out_WADI_$i.log
done

for i in {4..5}
do
    python main.py --model iTransformer --dataset creditcard_normal -w 10 -s 1 -d 2 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain --less > iTransformer_new/out_creditcard_normal_$i.log
    python main.py --model iTransformer --dataset creditcard -w 10 -s 1 -d 2 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain --less > iTransformer_new/out_creditcard_$i.log
    python main.py --model iTransformer --dataset GECCO -w 12 -s 1 -d 2 -e $epochs --feats -1 --k $i --loss softdtw --name rep_$i --retrain > iTransformer_new/out_GECCO_$i.log
done

for i in {1..5}
do
    python main.py --model iTransformer --dataset creditcard_normal -w 10 -s 1 -d 2 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain --less > iTransformer_new/out_creditcard_normal_norm$i.log
    python main.py --model iTransformer --dataset creditcard -w 10 -s 1 -d 2 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain --less > iTransformer_new/out_creditcard_norm$i.log
    python main.py --model iTransformer --dataset GECCO_normal -w 12 -s 1 -d 2 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain > iTransformer_new/out_GECCO_normal_norm$i.log
    python main.py --model iTransformer --dataset GECCO -w 12 -s 1 -d 2 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain > iTransformer_new/out_GECCO_norm$i.log
    python main.py --model iTransformer --dataset IEEECIS_new2.2 -w 10 -s 1 -d 10 -e $epochs --feats 30 --k $i --loss softdtw_norm --name rep_$i --retrain > iTransformer_new/out_IEEECIS_norm$i.log
    python main.py --model iTransformer --dataset MSL_new -w 96 -s 10 -d 96 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain --less > iTransformer_new/out_MSL_norm$i.log
    python main.py --model iTransformer --dataset SMAP_new -w 15 -s 2 -d 15 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain > iTransformer_new/out_SMAP_norm$i.log
    python main.py --model iTransformer --dataset SMD -w 350 -s 35 -d 70 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain > iTransformer_new/out_SMD_norm$i.log
    python main.py --model iTransformer --dataset SWaT_small -w 51 -s 5 -d 10 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain > iTransformer_new/out_SWaT_small_norm$i.log
    python main.py --model iTransformer --dataset SWaT_1D -w 10 -s 5 -d 2 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain > iTransformer_new/out_SWaT_1D_norm$i.log
    python main.py --model iTransformer --dataset UCR -w 10 -s 5 -d 2 -e $epochs --feats -1 --k $i --loss softdtw_norm --name rep_$i --retrain > iTransformer_new/out_UCR_norm$i.log
    python main.py --model iTransformer --dataset WADI -w 44 -s 4 -d 44 -e $epochs --feats 30 --k $i --loss softdtw_norm --name rep_$i --retrain --less > iTransformer_new/out_WADI_norm$i.log
done