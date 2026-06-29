# Plotting `pos_codes_comparison_accuracy.png` in greater detail

## Context

Previously, we compared the adaptive relational bottleneck without positional codes to the adaptive relational bottleneck with positional codes and the self attention mechanism with positional codes. We saw that the adaptive relational bottleneck without positional codes performed better on average. However, it is unclear if the other models with positional codes can also learn the task, but there is greater variability depending on the random seed.

## Actionable

Let's modify the figures in `scripts/context_identification/pos_codes/plot_pos_codes_comparison_results.py` to also included trajectories of individual models.

## Results

Just as I suspected. When we plot the individual trajectories, we see that some models with positional codes do indeed learn the task, but so many of them fail. However, without the positional codes, nearly all of the adaptive relational bottleneck models learn the task. We can conclude that there is something about the learning dynamics or loss landscapes that changes with positional codes.
