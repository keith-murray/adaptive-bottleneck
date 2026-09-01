# Understanding the effect of dataset size on the learnability of the task

## Context

For a while, we have been perplexed by the positional code results as documented in the table in the `README.md`. Namely, why would positional codes hurt the ARCA's ability to solve the task? Why do some seeds for the cross attention mechanism with positional codes solve the task while others don't?

In the last session, we got a hint at what the answer might be: we have been severely undersampling from the task. The `scripts/context_identification/pos_codes/full_batch_combinatorics.py` script revealed that there are millions to billions of possible combinations, and we have only been sampling 500 or so. Given this, it's incredible how well ACRA is able to generalize, indicating at how sample efficient it is.

## Task 1

Let's perform a sweep over number of samples for the various models and task labels (features and position). We want $L$ to be relatively manageable so that we can eventually have all samples in the training set. Let's have $L=8$ so that there are $13969440$ unique samples. We want to sweep sampling percentages logarithmically as well as by increments of $10\%$ (two separate sweep schedules). Let's have 10 seeds per condition. Also let's go back to mini-batching (as opposed to training on full batches) and set batch sizes at 64.

Let's reserve some amount of data as the test set. Since we can enumerate all the samples, let's save $20\%$ of the data for testing. In the case of the sampling percentages at $90\%$, just use the remaining $10\%$ as the test set. In the case of when all of the data is used for training, omit the test set.

We are also interested in the rate at which learning converges. We will be saving the data on a cluster, so memory constraints are not a factor, simply save all the data.

We will need to discuss how to configure the resulting slurm job and how many resources we should request.
