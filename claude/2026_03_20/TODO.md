# Training the Adaptable Bottleneck on the XOR task

## Background

At this point, we've shown that the adaptable bottleneck can extend the attention mechanism to handle a task were self-attention or relational-cross attention fails. This task was the `ContextIdentification` task whereby the model was given a context and a query and was tasked to identify the item in context that is closest to the query. The catch was that the label of the item in context could be the position in context or the identity of the feature. Self-attention could return the identity but not the position, and relational-cross attention could return the position but not the identity. The adaptive bottleneck could effectively learn both types of labels for the `ContextIdentification` task.

## XOR

The `ContextIdentification` task was informative in showing that the adaptive bottleneck can accommodate different types of target labels, but we should find a task that goes beyond labeling. Let's implement the XOR task in the style of the previous tasks we've implemented and train the adaptable bottleneck on it for various fixed values of $\gamma$ and for a learnable $\gamma$.

## XOR format

The inputs should be two length two vectors. For example, "false" could be [1,0] and "true" could be [0,1]. Therefore, an input could be [[1,0], [1,0]] which corresponds to two "false" values, which fails in XOR.
