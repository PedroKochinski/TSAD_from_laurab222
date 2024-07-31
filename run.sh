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

for i in 0.25 0.1 0.075 0.05 0.5 1e-2 1e-3 1e-4 5e-5 1e-5 5e-6 1e-6 1e-7
do 
    echo "detection level: $i"
    python main.py --model iTransformer --n_window 10 --dataset SMAP_new --q $i  
    # --test --checkpoint /Users/lauraboggia/VSCode_projects/TranAD/studies2.2/iTransformer_SMAP_new/detectionlvl_1e-05/checkpoints
    python main.py --model TranAD --n_window 10 --dataset SMAP_new --q $i 
    # --test --checkpoint /Users/lauraboggia/VSCode_projects/TranAD/studies2.2/iTransformer_SMAP_new/detectionlvl_1e-05/checkpoints
    
done
