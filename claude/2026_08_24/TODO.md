# Affect of weight scale on variability of learned solutions (part 2)

## Context

From the last session, we can see that the weight initialization has an affect on the positional task with smaller weights leading to a higher chance of convergence for all networks. However, in speaking with my postdoc mentor, we want to move away from SGD for this analysis and move towards full batch gradient descent.

## Full batch gradient descent

This should be fairly straight forward, but one thing we will need to be mindful of is the combinatorics of the task. We can vary $L$ the number of classes (and feature vector length) to make the combinatorics more manageable.

### First step

Let's just see the functional relationship between $L$ and the number of unique task samples.

### Second step

Let's implement training on the full batch gradient descent, where now the x axis of the training plot will just be training steps, not epochs as epochs implies mini-batching.

### Third step

Let's implement the sweep across weight initialization scales for both tasks, just like in `scripts/context_identification/pos_codes/weight_scale_pos_codes_sweep.py`.
