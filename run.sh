#!/bin/bash
# source /cvmfs/sft.cern.ch/lcg/views/LCG_105_cuda/x86_64-el9-gcc11-opt/setup.sh

# for (( i=10; i<100; i+=10 ))
# do
#     echo "Window size: $i"
#     python main.py --model iTransformer --n_window $i --dataset SMAP --retrain
# done

# for (( i=100; i<1000; i+=200 ))
# do
#     echo "Window size: $i"
#     python main.py --model iTransformer --n_window $i --dataset SMAP --retrain
# done

# for i in 100 # 8500 200 500 1000 2000
# do
#     echo "Window size: $i"
#     python main.py --model iTransformer --n_window $i --dataset SMAP --retrain
# done

# for i in 0.25 0.1 0.075 0.05 0.5 1e-2 1e-3 1e-4 5e-5 1e-5 5e-6 1e-6 1e-7
# do 
#     echo "detection level: $i"
#     python main.py --model iTransformer --n_window 10 --dataset SMAP_new --q $i  
#     # --test --checkpoint /Users/lauraboggia/VSCode_projects/TranAD/studies2.2/iTransformer_SMAP_new/detectionlvl_1e-05/checkpoints
#     python main.py --model TranAD --n_window 10 --dataset SMAP_new --q $i 
#     # --test --checkpoint /Users/lauraboggia/VSCode_projects/TranAD/studies2.2/iTransformer_SMAP_new/detectionlvl_1e-05/checkpoints
    
# done

# for (( i=1; i<5; i++ ))
# do 
# # for f in 40  # 50 75 100 150
# # do
#     echo "$i : rep"
#     python main.py --model iTransformer --n_window 100 --dataset GECCO --step_size 50 --feats -1 --retrain --name $i
    
# # done
# done

# for e in 10 15 20
# do
#     echo "# features $f"
#     python main.py --model iTransformer --n_window 100 --dataset IEEECIS_pca_scaled --steps_size 50 --feats 100 --epochs $e --retrain
    
# done

# for n in 100 200 500 1000 2000
# do
#     s=$(($n/2))
#     echo "n_window $n step_size $s"
#     python main.py --model iTransformer --n_window $n --dataset IEEECIS_pca_scaled --step_size $s --retrain --feats 40
# done

# for n in 5 10 20 30
# do
#     echo "n_window $n"
#     python main.py --model iTransformer --n_window $n --dataset ATLAS_TS --step_size 1 --retrain --less
#     python main.py --model TranAD --n_window $n --dataset ATLAS_TS --step_size 1 --retrain --less
# done

# for m in OmniAnomaly DAGMM iTransformer TranAD  #  LSTM_AD MAD_GAN
# do
#     echo "Model $m"
#     python main.py --model $m --n_window 10 --dataset GECCO --step_size 1 --retrain --feats -1
# done

# echo "data set IEEECIS_new2.2"
# for i in {1..5}
# do
#     echo "rep $i"
#     # echo "data set $d"
#     python main.py --model TranAD --n_window 10 --dataset IEEECIS_new2.2 --step_size 1 --epochs 200 --retrain --feats 30 --k $i --name $i
#     # python main.py --model iTransformer --n_window 10 --dataset IEEECIS_new2.2 --step_size 1 --epochs 200 --retrain --feats 30 --k $i --name latent2_$i
#     # python main.py --model iTransformer --n_window 100 --dataset IEEECIS_new2.2 --step_size 50 --epochs 200 --retrain --feats 30 --k $i --name latent2_$i
# done

# for d in creditcard_normal # UCR GECCO SWaT_1D # ATLAS_TS
# do
#     echo "data set $d"
#     for i in {1..5}
#     do  
#         echo "rep $i"
#         python main.py --model TranAD --n_window 10 --dataset $d --step_size 1 --epochs 200 --retrain --feats -1 --k $i --name $i
#         python main.py --model iTransformer --n_window 10 --dataset $d --step_size 1 --epochs 100 --retrain --feats -1 --k $i --name latent2_$i 
#         # python main.py --model iTransformer --n_window 100 --dataset $d --step_size 50 --epochs 100 --retrain --feats -1 --k $i --name latent2_$i 
#     done
# done

# for d in SMD # SWaT UCR GECCO ATLAS_TS SMD MSL_new
# do
#     echo "data set $d"
#     for i in {1..5}
#     do  
#         echo "rep $i"
#         # python main.py --model TranAD --n_window 10 --dataset $d --step_size 1 --epochs 200 --test --feats -1 --k $i --name $i --less
#         python main.py --model iTransformer --n_window 10 --dataset $d --step_size 1 --epochs 200 --test --feats -1 --k $i --name latent2_$i --less
#         python main.py --model iTransformer --n_window 100 --dataset $d --step_size 50 --epochs 200 --test --feats -1 --k $i --name latent2_$i --less
#     done
# done

# # for d in  # SMD SMAP_new UCR ATLAS_TS SWaT
# do
#     echo "data set $d"
#     python main.py --model USAD --n_window 10 --dataset $d --step_size 1 --retrain --feats -1 
# done

# for d in SMD SMAP_new UCR ATLAS_TS
# do
#     echo "data set $d"
#     python main.py --model MERLIN --n_window 10 --dataset $d --step_size 1 --retrain --feats -1 
# done

# python main.py --model TranAD --n_window 10 --dataset creditcard_normal --step_size 1 --epochs 5 --retrain --feats -1 --k -1 --name test
python main.py --model iTransformer --n_window 10 --dataset WADI --step_size 1 --epochs 5 --retrain --feats 30 --k -1 --name latent2_test
python main.py --model iTransformer --n_window 100 --dataset WADI --step_size 50 --epochs 5 --retrain --feats 30 --k -1 --name latent2_test

# python main.py --model iTransformer --n_window 2000 --dataset ATLAS_DQM_TS --retrain --step_size 1000 --feats -1 --name train_all
# python main.py --model iTransformer --n_window 1000 --dataset WADI --retrain --step_size 500 --feats -1 --name latent2_1 --k 1