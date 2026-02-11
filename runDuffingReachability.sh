#!/bin/bash

python scripts/reachabilityDuffing.py --train-ratio 0.2 --model mamba 
python scripts/reachabilityDuffing.py --train-ratio 0.8 --model mamba 
python scripts/reachabilityDuffing.py --train-ratio 0.2 --model mamba --ood
python scripts/reachabilityDuffing.py --train-ratio 0.8 --model mamba --ood

python scripts/reachabilityDuffing.py --train-ratio 0.2 --model lstm 
python scripts/reachabilityDuffing.py --train-ratio 0.8 --model lstm 
python scripts/reachabilityDuffing.py --train-ratio 0.2 --model lstm --ood
python scripts/reachabilityDuffing.py --train-ratio 0.8 --model lstm --ood