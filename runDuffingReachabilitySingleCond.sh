#!/bin/bash

# default parameters for dataset generation
DV=0.3
DV_ood=$(awk "BEGIN {print $DV*2}")

echo "Using delta-v radius: ${DV}"
echo "Using OOD delta-v radius: ${DV_ood}"

# run dataset generation for the Duffing system if files do not exist
if [ ! -f "data/test/duffing_single_monte_carlo_trajectories_dv_${DV}_dt_0.02_n_20000.npz" ] || [ ! -f "data/test/duffing_single_monte_carlo_trajectories_dv_${DV_ood}_dt_0.02_n_20000_ood.npz" ]; then
    echo "Generating dataset for the Duffing system..."
    python scripts/datagen/duffing_single_delta_v_gen.py --dv ${DV} --dt 0.02 --n 20000 --ood
    echo "Dataset generation complete."
else
    echo "Dataset already exists. Skipping generation."
fi



python scripts/reachabilityDuffingSingle.py --train-ratio 0.8 --model mamba --dv ${DV} --lookback 4 --train-timesteps 5 --pdf
python scripts/reachabilityDuffingSingle.py --train-ratio 0.8 --model lstm --dv ${DV} --lookback 4 --train-timesteps 5 --pdf

# train timesteps 50 = 1 second window
python scripts/reachabilityDuffingSingle.py --train-ratio 0.8 --model mamba --dv ${DV} --lookback 10 --train-timesteps 50 --pdf
python scripts/reachabilityDuffingSingle.py --train-ratio 0.8 --model lstm --dv ${DV} --lookback 10 --train-timesteps 50 --pdf

python scripts/reachabilityDuffingSingle.py --train-ratio 0.8 --model mamba --dv ${DV} --lookback 10 --train-timesteps 100 --pdf
python scripts/reachabilityDuffingSingle.py --train-ratio 0.8 --model lstm --dv ${DV} --lookback 10 --train-timesteps 100 --pdf