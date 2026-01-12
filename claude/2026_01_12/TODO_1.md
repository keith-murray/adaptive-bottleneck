# Allowing $\gamma$ to be learned

## Preamble

At this point, we have validated that our adaptable bottleneck can be tuned to meet the needs of specific tasks. Now what we want to do is to allow $\gamma$ to be learned via backpropigation. Refer to the `script/block_task/online_opt_gamma.py` script to get a sense as to how this can be done, although it might be out of date.

## Actionables

For each of the `ContextIdentification` task label modes, 'feature', 'position', and 'both', we want to $\gamma$ to be learned. We will perform a sweep across seeds with $\gamma$ being initialized at different locations.

I'm thinking of a similar setup to the `scripts/context_identification/context_identification_bottleneck.py` script, although let's only focus on "softmax" attention and ignore "linear" attention. $\gamma$ should be initialized with a value from $\gamma\in\{0.0, 0.25, 0.5, 0.75, 1.0\}$. For each initialization, there should b five different seeds.
