# Cleaning up figures for poster

## Context

I am making a poster about the work done in this repository. There are three figures I would like to use, but they need to be cleaned up for the poster.

## First pass

The first figure is `results/context_identification/fixed_gamma/test_acc_dynamics_context_identification.png`. Remove the super title and have the column titles just be "feature label" and "positional label".

The second figure is `results/context_identification/online_gamma/online_gamma_20seeds_combined_zoom.png`. Make the figure 1 row by 2 columns. Both of the panels should have the same y axes. Note that each panel has a left and a right y axis. In the first panel, label the left axis and in the second panel label the right axis. There should be column titles that are just "feature label" and "positional label".

The third figure is `results/context_identification/pos_codes/pos_codes_comparison_accuracy.png`. Make this figure 3 rows by 2 columns. Remove the super title and have the column titles just be "feature label" and "positional label". Label the rows accordingly since each row is a different model.

The pattern is that each figure should have two columns: the left column is feature labels and the right column is positional labels. No figure should have a super title, that will be in the caption.

## Second pass

The first figure mostly looks good. Let's make it a similar size as `results/context_identification/online_gamma/online_gamma_20seeds_combined_zoom.png` for consistency.

In the second figure, remove the right y axis ticks in the left panel.

In the third figure, reduce the size of the y axis labels.

Across all figures, let's change the column titles to "Feature label" and "Positional label". This is just a capitalization error on my part.

## Third pass

The last fix is the third figure. Upon putting it into the poster, I realized we have new names for the different models we tested. Here are the names

- Adaptive (no pos. codes) -> ARCA
- Adaptive + pos. codes -> ARCA (pos. codes)
- Attention + pos. codes -> Cross attention (pos. codes)
