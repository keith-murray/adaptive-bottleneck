# Submitting the sampling percentage sweep job

## Context

In the last session, we built a sweep across dataset percentages to see if the ARCA mechanism has a sampling efficiency benefit compared to ARCA with positional codes and cross attention with positional codes. We submitted the profiling job and saw that it was taking too long. We implemented a fix, and resubmitted the profiling job. The log for the new profiling job is in `slurm/logs/sampling_pct_pos_codes_profile_5995808.out`.

## Task 1: Submitting the sweep

Let's now configure what the parameters from the slurm job should be. One thing to keep in mind is that we have a max CPU usage of 250 cores at one time. To have the job complete as soon as possible, we should be using as many of the 250 cores at one time as possible.

## Task 2: Plotting the results

The slurm job submitted in the previous task has successfully ran. The resulting data is 7 GB, so I won't pull the data to my local computer. Let's design some figures to plot the results.

First, let's make four separate figures:

1. figure that plots the distribution of final training accuracy for the linear sweep across both output labels and all models
2. figure that plots the distribution of final testing accuracy for the linear sweep across both output labels and all models
3. figure that plots the distribution of final training accuracy for the log sweep across both output labels and all models
4. figure that plots the distribution of final testing accuracy for the log sweep across both output labels and all models

Each panel should be a strip plot where there is one type of label output and all models. On the x-axis is the percent of data used to train the model, with different strips corresponding to different models in each x-axis "bin". Each model should have the same color across all the different percentages. The y-axis is of course the final training/testing accuracy.

Let's make the plotting script such that I can send it to the cluster and run it on the data there.
