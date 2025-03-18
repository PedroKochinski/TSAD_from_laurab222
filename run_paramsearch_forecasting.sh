#!/bin/bash

# source /eos/user/l/lboggia/miniforge3/bin/activate
# conda activate /eos/user/l/lboggia/miniforge3/envs/test4

# cd /eos/user/l/lboggia/itransformerad  #  FOR FORECASTING NOT RECONSTRUCTION

kfold=-1
epochs=10
W_default=96
S_default=1
feats=-1


for d in creditcard_normal # creditcard 
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=50
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
    done
done

for d in IEEECIS_new2.2
do
    feats_less=30
    echo "data set $d"
    for i in {1..5}
    do
        echo "rep $i"
        W_plus=50
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d $W_plus -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 10 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
    done
done

for d in GECCO_normal # GECCO 
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=50
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        W_minus=12 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
    done
done


for d in MSL_new 
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=250
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 50 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
    done
done

for d in SMAP_new 
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=850
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 170 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        W_minus=15 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 3 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
    done
done

for d in SMD 
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=350
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 70 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
    done
done

for d in SWaT 
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=1600
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 51 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        W_minus=51
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
    done
done

for d in SWaT_1D
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=50
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
    done
done

for d in UCR
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=100
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 20 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain --forecasting
    done
done

for d in WADI 
do
    feats_less=30
    echo "data set $d"
    for i in {1..5}
    do
        echo "rep $i"
        W_plus=750
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 256 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_plus -s $S_default -d 150 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        W_minus=44
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d $W_minus -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_minus -s $S_default -d 9 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d $W_default -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
        python main.py --model iTransformer --dataset $d -w $W_default -s $S_default -d 19 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain --forecasting
    done
done
