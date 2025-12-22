# Adaptive Relational Bottleneck to Improve Abstract Learning

This repo contains code and analyses pertaining to project about understanding what tasks the relational bottleneck should be used for. It is currently unclear when the relational bottleneck should be used over self-attention, so in this project, we developed a model that interpolates between the two approaches via
$$
\operatorname{softmax}\left( \frac{(XW_q)(XW_k)^\bot}{d_k} \right)((1-\gamma)X+\gamma I)W_v
$$
where the value $((1-\gamma)X+\gamma I)W_v$ is now controlled by $\gamma$. A lower $\gamma$ value corresponds to more self-attention and a higher $\gamma$ value corresponds to more relational bottleneck.
