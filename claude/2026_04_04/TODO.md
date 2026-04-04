# Training ARCA on XOR task

## Background

We've already implemented the XOR task, but in preparation for a Cognitive Computational Neuroscience abstract submission, we had to go back through the results for the Context Identification task. We found that the best learning algorithm for ARCA to perform both variants of the Context Identification task was an online update of the gamma parameter with stochastic gradient descent (SGD) with the gamma parameter initialized at $\gamma_0=0.5$. Note that the learning algorithm for gamma needed to be SGD while the algorithm for the weights of the networks could be ADAM, and that the initial $\gamma_0$ needed to be $0.5$, otherwise the model would be stuck in a local minimum. Refer to the `scripts/context_identification/online_gamma/online_gamma_20seeds_sweep.py` script for more details.

## XOR task

We know that RCA and ARCA with $\gamma=1$ are uniquely able to solve the XOR task, but we don't know if this new initialization and learning regime (online with SGD for $\gamma$) will allow ARCA to learn to the XOR task.

I want a script much like the `scripts/context_identification/online_gamma/online_gamma_20seeds_sweep.py` script, but for XOR. This new script should have three different models:

1. ARCA with $\gamma$ fixed to $1$. Hence just RCA.
2. ARCA with $\gamma$ fixed to $0$. Hence just cross attention.
3. ARCA with $\gamma_0=0.5$ and updated via SGD.

I then want to plot the usual results for testing accuracy over epochs and gamma over epochs.
