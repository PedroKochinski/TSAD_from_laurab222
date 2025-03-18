#!/bin/bash


# source /eos/user/l/lboggia/miniforge3/bin/activate
# conda activate /eos/user/l/lboggia/miniforge3/envs/test4

# cd /eos/user/l/lboggia/VSCode_projects/iTransformerAD/itransformerad

kfold=-1
epochs=10
W_default=96
feats=-1


for d in creditcard_normal # creditcard 
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=50
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 5 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 5 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
    done
done

for d in IEEECIS_new2.2
do
    feats_less=30
    echo "data set $d"
    for i in {1..5}
    do
        W_plus=50
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 5 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 5 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
    done
done

for d in GECCO_normal # GECCO 
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=50
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 5 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 5 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=12
        python main.py --model iTransformer --dataset $d -w $W_minus -s 6 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 6 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
    done
done

for d in MSL_new
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=250
        python main.py --model iTransformer --dataset $d -w $W_plus -s 125 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 125 -d 50 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d 50 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
    done
done

for d in SMAP_new
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=850
        python main.py --model iTransformer --dataset $d -w $W_plus -s 425 -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 425 -d 170 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 85 -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 85 -d 170 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=15
        python main.py --model iTransformer --dataset $d -w $W_minus -s 8 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 8 -d 3 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 2 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 2 -d 3 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
    done
done

for d in SMD
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=350
        python main.py --model iTransformer --dataset $d -w $W_plus -s 175 -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 175 -d 70 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 35 -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 35 -d 70 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
    done
done

for d in SWaT
do
    echo "data set $d"
    for i in {1..5}
    do  
        echo "rep $i"
        W_plus=1600
        python main.py --model iTransformer --dataset $d -w $W_plus -s 800 -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 800 -d 51 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 160 -d 256 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 160 -d 51 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=51
        python main.py --model iTransformer --dataset $d -w $W_minus -s 26 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 26 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
    done
done

for d in SWaT_1D
do
    echo "data set $d"
    for i in {1..5}
    do 
        echo "rep $i"
        W_plus=50
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 25 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 5 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 5 -d 10 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
    done
done

for d in UCR
do
    echo "data set $d"
    for i in {1..5}
    do
        echo "rep $i"
        W_plus=100
        python main.py --model iTransformer --dataset $d -w $W_plus -s 50 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 50 -d 20 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 10 -d $W_plus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 10 -d 20 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        W_minus=10 
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 5 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d $W_minus -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 1 -d 2 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats --k $kfold --name rep_$i --retrain
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
        python main.py --model iTransformer --dataset $d -w $W_plus -s 375 -d 256 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 375 -d 150 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 75 -d 256 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_plus -s 75 -d 150 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        W_minus=44 
        python main.py --model iTransformer --dataset $d -w $W_minus -s 22 -d $W_minus -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 22 -d 9 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 4 -d $W_minus -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_minus -s 4 -d 9 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        # W_default
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d $W_default -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 48 -d 19 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d $W_default -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
        python main.py --model iTransformer --dataset $d -w $W_default -s 10 -d 19 -e $epochs --feats $feats_less --k $kfold --name rep_$i --retrain
    done
done