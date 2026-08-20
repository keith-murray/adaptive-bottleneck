# Affect of weight scale on variability of learned solutions

## Context

Following from the previous sessions work (which was almost a month ago), we see that there is a lot of variability in the learning trajectories of ARCA with positional codes and cross attention with positional codes, but all seeds converge to an optimal solution for ARCA without positional codes. Why is this? We know that ARCA with pos. codes and cross attention with pos. codes. can both solve the task, so what is the difficulty?

## Idea

One idea is that there is competing gradient information from the positional code weights. A way to test this is to initialize weights with a very small scale, something like $g\in \{10^{-8}, 10^{-7}, 10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ where $g$ is the scale of the initialized weights.

## Testing

We want to test this idea of scale across all three models and many trajectories. As the initialized weights get smaller, we will have train for more epochs. The final figure should be 6 panels, one for each model and task condition, much like in `results/context_identification/pos_codes/pos_codes_comparison_accuracy.png` but we want to have scale of the initialized weights on the x-axis and test accuracy on the y-axis. The panels should be jitter plots with the means visualized.

We will also be running this large sweep on an HPC. There is an example of the slurm/bash script for the HPC in the new `slurm/` directory.
