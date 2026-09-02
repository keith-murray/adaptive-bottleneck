#!/bin/bash
#SBATCH --job-name=sampling_plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/sampling_pct_pos_codes_plot.out

cd "$HOME/adaptive-bottleneck"

source /usr/people/km3199/.local/bin/env

uv run python scripts/context_identification/pos_codes/plot_sampling_pct_pos_codes_results.py
