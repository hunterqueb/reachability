#!/bin/bash

# default parameters for dataset generation
DV=2.0

# short term training, mamba excels here
python scripts/reachability2BP_TC.py --train-ratio 0.8 --model mamba --dv ${DV} --pdf --horizon 3
python scripts/temporal_Reachability2BP_TC.py --train-ratio 0.8 --model lstm --dv ${DV} --train-timesteps 5 --lookback 4 --pdf


# longer term training, shows LSTM at best
python scripts/reachability2BP_TC.py --train-ratio 0.8 --model mamba --dv ${DV} --pdf --horizon 10
python scripts/temporal_Reachability2BP_TC.py --train-ratio 0.8 --model lstm --dv ${DV} --train-timesteps 20 --lookback 4 --pdf
# python scripts/temporal_Reachability2BP_TC.py --train-ratio 0.8 --model lstm --dv ${DV} --eps