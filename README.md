# The adaptive relational bottleneck: regulation of attention in the service of abstraction

This repo contains code and analyses pertaining to a project about understanding when it is appropriate to use the relational bottleneck. Given that the relational bottleneck is an inductive bias, the No Free Lunch Theorem tells us that there are some tasks that using the relational bottleneck will result in worse performance. Therefore, if we allow the relational bottleneck to be adaptive, we might be able to recover performance on those tasks.

## Architecture

### Previous work

As proposed in Vaswani et al. (2017), the attention mechanism is formalized as
$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$
where matrices $Q,K,V$ correspond to concatenated query, key, and value vectors. To explicitly operate over data, the attention mechanism is extended to the cross-attention mechanism as
$$
\operatorname{Cross}\operatorname{Attention}(X_q, X_c; W_q, W_k, W_v) = \operatorname{softmax}\!\left(\frac{(X_qW_q)(X_cW_k)^\top}{\sqrt{d}}\right)(X_cW_v)
$$
where some query matrix $X_q$ is compared across a contextual matrix $X_c$ to index similar items in the context as determined by the query and key matrices $W_q,W_k$.

Following from the work in Altabaa et al. (2024), the cross-attention mechanism is extended to explicitly compute relations to relational cross attention (RCA) mechanism to operate over abstract symbols via
$$
\operatorname{RCA}(X_q, X_c; W_q, W_k, S)= \operatorname{softmax}\!\left(\frac{(X_qW_q)(X_cW_k)^\top}{\sqrt{d}}\right)S~.
$$
Here, values $V$ are now symbols $S$ that are, unlike in cross attention, restricted from seeing the context $X_c$. This is an instantiation of the relational bottleneck principle (Webb et al., 2024).

### Our contribution

We extend the cross- and relational-attention mechanisms in the spirit of the relational bottleneck principle by formulating the _adaptive relational-cross attention_ (ARCA) mechanism as
$$
\operatorname{ARCA}(X_q, X_c; \gamma, W_q, W_k, W_v)= \operatorname{softmax}\!\left(\frac{(X_qW_q)(X_cW_k)^\top}{\sqrt{d}}\right)\left[(1-\gamma)X_c+\gamma I\right]W_v
$$
whereby the value matrix $V$ now becomes
$$
V = \left[(1-\gamma)X_c+\gamma I\right]W_v~,
$$
effectively acting as a linear interpolation between the cross- and relational-cross attention mechanisms. The $\gamma$ term can be a free parameter, allowing for gradient descent to ''open'' or ''close'' the relational bottleneck depending on task demands. When $\gamma=1$, ARCA collapses to RCA where $S=\gamma I W_v$, and when $\gamma=0$, ARCA to cross attention.

## Results

### Context Identification

To demonstrate the computation benefits of ARCA in a simple setting, we designed the _contextual identification task_ whereby a model is presented with a context matrix $X_c$ and a query vector $x_q$ and is tasked to identify which item the context matrix $X_c$, denoted as $x^*_c$, is most similar to the query $x_q$. Input data consisted of one-hot vectors, with a distance metric defined as the absolute value of the difference between the one-hot index of two vectors $\mathbf{u}, \mathbf{v}$
$$
    d(\mathbf{u}, \mathbf{v}) = \left| \arg\max_i \, u_i - \arg\max_j \, v_j \right|~.
$$
Crucially, we manipulated the task such that the model was task to either output the position of $x^*_c$ in the context matrix $X_c$, termed the _positional_ label, or the feature identity of $x^*_c$, termed the _feature_ label.

Given the positional nature of the  context identification task, we thought to test whether ARCA provided a benefit over injecting positional codes directly into the context matrix $X_c$, much like it Vaswani et al. 2017. We encoded position as a one-hot vector and appended each row in $X_c$ with its corresponding vector identity. Our results indicate that ARCA without positional codes performed better than both ARCA with positional codes and cross attention with positional codes.

| Condition.                 | Feature         | Position        |
| --------                   | --------        | --------        |
| ARCA (no Pos)              | $98.5\pm 0.4$   | $99.0\pm 0.2$   |
| ARCA (with Pos)            | $79.9\pm 5.2$   | $80.7\pm 5.7$   |
| Cross attention (with Pos) | $78.0\pm 5.1$   | $72.3\pm 7.9$   |
