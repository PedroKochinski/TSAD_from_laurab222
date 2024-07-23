#!/bin/bash

source "/Users/lauraboggia/VSCode_projects/TranAD/.conda/activate"
conda activate "/Users/lauraboggia/VSCode_projects/TranAD/.conda"

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

for i in 1e-3 1e-4 5e-5 1e-5 5e-6 1e-6 1e-7
do 
    echo "detection level: $i"
    python main.py --model iTransformer --n_window 10 --q $i --dataset SMAP --retrain 
done
