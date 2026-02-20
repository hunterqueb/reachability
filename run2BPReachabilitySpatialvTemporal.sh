#!/bin/bash

# default parameters for dataset generation
DV=2.0

python scripts/reachability2BP_TC.py --train-ratio 0.8 --model mamba --dv ${DV}

python scripts/temporal_Reachability2BP_TC.py --train-ratio 0.8 --model lstm --dv ${DV} --train-timesteps 5 --lookback 4
# python scripts/temporal_Reachability2BP_TC.py --train-ratio 0.8 --model lstm --dv ${DV}