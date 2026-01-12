# Validating a half-open bottleneck

## Preamble

At this point we have validated the adaptable bottleneck attention mechanism on the `ContextIdentification` task. Specifically:

- When the `label_mode` is 'feature', models with $\gamma\sim 0$ uniquely perform well.
When the `label_mode` is 'position', models with $\gamma\sim 1$ uniquely perfrom well.

Now, we want to show that there are some tasks where we would like a bottleneck that is half open. I.e. a task where a model needs $\gamma\sim 0.5$. We call these tasks "partially-relational tasks".

## A partially-relational task

There are a few tasks that we could use, but the simplest would be an extension of the `ContextIdentification` task where instead of asking for only 'feature' or 'position', we ask for both.

Now this doesn't mean that we should create a single target that is a combination of feature and positional information, but that the model should have two outputs for both targets.

## Extensions to the model

Here is the adaptable bottleneck attention mechanism:
$$
\text{softmax}\left( \frac{(X_qW_q)(X_cW_k)^\top}{\sqrt{d}} \right)\left[(1-\gamma)X_c+\gamma I\right]W_v
$$
Let's say that the output of the mechanism is $Z\in\mathbb{R}^{1\times d}$. We can add two different decoders $W_f,W_p\in\mathbb{R}^{d\times d}$ where feature information would be readout as $ZW_f$ and positional information would be readout as $ZW_p$.

To optimize, our loss function would simply be a sum of the cross entropy of both $ZW_f$ and $ZW_p$:
$$
\mathcal{L}=\operatorname{CE}(ZW_f, f)+\operatorname{CE}(ZW_p, p)
$$
where $f$ is the integer of the feature and $p$ is the interger of the position.
