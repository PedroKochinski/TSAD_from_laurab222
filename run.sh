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

kfold=-1
epochs=10
feats=-1

for i in 1
do
    # python main.py --model iTransformer --dataset creditcard_normal -w 10 -s 1 -d 2 -e $epochs --feats $feats\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_creditcard_normal/window10_steps1_dmodel2_feats-1_eps10_Huber/rep_$i/checkpoints/ > iTransformer/creditcard_normal_Huber_MSEtest$i.log
    python main.py --model iTransformer --dataset GECCO_normal -w 12 -s 1 -d 2 -e $epochs --feats $feats\
    --loss Huber_quant --k $i --name IQRloss_rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_GECCO_normal/window12_steps1_dmodel2_feats-1_eps10_Huber_quant/rep_$i/checkpoints/ # > iTransformer/GECCO_normal_Huber_quant_MSEtest$i.log
    # python main.py --model iTransformer --dataset IEEECIS_new2.2 -w 10 -s 1 -d 2 -e $epochs --feats 30\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_IEEECIS_new2.2/window10_steps1_dmodel2_feats30_eps10_Huber/rep_$i/checkpoints/ > iTransformer/IEEECIS_new2.2_Huber_MSEtest$i.log
    # python main.py --model iTransformer --dataset MSL_new -w 96 -s 10 -d 96 -e $epochs --feats $feats\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_MSL_new/window96_steps10_dmodel96_feats-1_eps10_Huber/rep_$i/checkpoints/ > iTransformer/MSL_new_Huber_MSEtest$i.log
    # python main.py --model iTransformer --dataset SMAP_new -w 15 -s 2 -d 15 -e $epochs --feats $feats\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_SMAP_new/window15_steps2_dmodel15_feats-1_eps10_Huber/rep_$i/checkpoints/ > iTransformer/SMAP_new_Huber_MSEtest$i.log
    # python main.py --model iTransformer --dataset SMD -w 350 -s 35 -d 70 -e $epochs --feats $feats\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_SMD/window350_steps35_dmodel70_feats-1_eps10_Huber/rep_$i/checkpoints/ > iTransformer/SMD_Huber_MSEtest$i.log
    # python main.py --model iTransformer --dataset SWaT -w 51 -s 5 -d 10 -e $epochs --feats $feats\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_SWaT/window51_steps5_dmodel10_feats-1_eps10_Huber/rep_$i/checkpoints/ > iTransformer/SWaT_Huber_MSEtest$i.log
    # python main.py --model iTransformer --dataset SWaT_1D -w 10 -s 5 -d 2 -e $epochs --feats $feats\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_SWaT_1D/window10_steps5_dmodel2_feats-1_eps10_Huber/rep_$i/checkpoints/ > iTransformer/SWaT_1D_Huber_MSEtest$i.log
    # python main.py --model iTransformer --dataset UCR -w 10 -s 5 -d 2 -e $epochs --feats $feats\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_UCR/window10_steps5_dmodel2_feats-1_eps10_Huber/rep_$i/checkpoints/ > iTransformer/UCR_Huber_MSEtest$i.log
    # python main.py --model iTransformer --dataset WADI -w 44 -s 4 -d 44 -e $epochs --feats 30\
    # --loss Huber --k $i --name testMSE/rep_$i --test --checkpoint iTransformer_loss_fct/iTransformer_WADI/window44_steps4_dmodel44_feats30_eps10_Huber/rep_$i/checkpoints/ > iTransformer/WADI_Huber_MSEtest$i.log
done