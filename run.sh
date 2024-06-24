#!/bin/bash

# source "/Users/lauraboggia/VSCode_projects/TranAD/.conda/activate"
# conda activate " /Users/lauraboggia/VSCode_projects/TranAD/.conda"

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

for i in 2000 8500 # 200 500 1000
do
    echo "Window size: $i"
    python main.py --model iTransformer --n_window $i --dataset SMAP --retrain
done
