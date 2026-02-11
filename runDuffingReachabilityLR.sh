#!/bin/bash

# run the reachability analysis for the Duffing system with different learning rates for lstm and mamba
# learning rates to test
lrs=(0.0001 0.001 0.01 0.1)

# loop over learning rates and run the reachability analysis for both models
# for lr in "${lrs[@]}"
# do
#     echo "Running reachability analysis for learning rate: $lr"
#     python3 scripts/reachabilityDuffing.py --train-ratio 0.8 --dv 0.6 --model lstm --lr $lr
#     python3 scripts/reachabilityDuffing.py --train-ratio 0.8 --dv 0.6 --model mamba --lr $lr
# done

for lr in "${lrs[@]}"
do
    echo "Running reachability analysis for learning rate: $lr"
    python3 scripts/reachabilityDuffing.py --train-ratio 0.0025 --dv 0.6 --model lstm --lr $lr
    python3 scripts/reachabilityDuffing.py --train-ratio 0.0025 --dv 0.6 --model mamba --lr $lr
done