import jax
from jax import random
import jax.numpy as jnp
import tensorflow as tf


class BlockComparison:
    """
    Block comparison task implemented as TensorFlow dataset wrapper.

    Each example is a set of `num_vectors` one-hot vectors of length `vector_len` (even),
    where "block" means first half vs second half of the vector.

    Label = 1.0 if all vectors' 1s lie in the same half (all-first or all-last),
    Label = 0.0 otherwise (at least one vector in a different half).

    Uses JAX PRNG for reproducibility and exports a tf.data.Dataset.
    """

    def __init__(self, key, num_samples=1000, num_vectors_and_vector_len=4):
        assert num_vectors_and_vector_len % 2 == 0, "vector length must be even"
        assert num_samples % 2 == 0, "num_samples must be even for balanced classes"

        self.key = key
        self.num_samples = num_samples
        self.num_vectors = num_vectors_and_vector_len
        self.vector_len = num_vectors_and_vector_len
        self.mid = self.vector_len // 2

    def generate_subkey(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def _one_hot(self, idx):
        return jnp.eye(self.vector_len, dtype=jnp.float32)[idx]

    def generate_same(self, key):
        """
        Generate one positive example: all vectors in the same block.
        Returns:
            X: (num_vectors, vector_len) float32
        """
        k_block, k_idxs = random.split(key)

        # same_block: 0 => first half, 1 => last half
        same_block = random.bernoulli(k_block, 0.5)

        # sample indices in the chosen half
        # (shape: (num_vectors,))
        idxs_first = random.randint(k_idxs, (self.num_vectors,), 0, self.mid)
        idxs_last  = random.randint(k_idxs, (self.num_vectors,), self.mid, self.vector_len)
        idxs = jnp.where(same_block, idxs_last, idxs_first)

        X = jnp.stack([self._one_hot(i) for i in idxs], axis=0)
        return X

    def generate_different(self, key):
        """
        Generate one negative example: mix blocks and ensure at least one mismatch.
        Returns:
            X: (num_vectors, vector_len) float32
        """
        k_blocks, k_idxs = random.split(key)

        # independent block assignment per vector
        blocks = random.bernoulli(k_blocks, 0.5, shape=(self.num_vectors,))  # 0 first, 1 last

        # pick indices for each vector conditional on its block
        idxs_first = random.randint(k_idxs, (self.num_vectors,), 0, self.mid)
        idxs_last  = random.randint(k_idxs, (self.num_vectors,), self.mid, self.vector_len)
        idxs = jnp.where(blocks, idxs_last, idxs_first)

        # ensure at least one mismatch:
        # if all blocks identical, flip the first one
        all_same = jnp.all(blocks == blocks[0])
        blocks = blocks.at[0].set(jnp.logical_not(blocks[0])) if all_same else blocks

        # recompute idx for first element if we flipped its block
        # (only matters in the all_same case)
        idx0_first = random.randint(k_idxs, (), 0, self.mid)
        idx0_last  = random.randint(k_idxs, (), self.mid, self.vector_len)
        idx0 = jnp.where(blocks[0], idx0_last, idx0_first)
        idxs = idxs.at[0].set(jnp.where(all_same, idx0, idxs[0]))

        X = jnp.stack([self._one_hot(i) for i in idxs], axis=0)
        return X

    def generate_jax_tensors(self):
        """
        Returns:
            X: (num_samples, num_vectors, vector_len)
            y: (num_samples,) float32
        """
        half = self.num_samples // 2

        # keys for each sample
        key_same = self.generate_subkey()
        key_diff = self.generate_subkey()
        keys_same = random.split(key_same, half)
        keys_diff = random.split(key_diff, half)

        X_same = jnp.stack([self.generate_same(k) for k in keys_same], axis=0)
        X_diff = jnp.stack([self.generate_different(k) for k in keys_diff], axis=0)

        y_same = jnp.ones((half,), dtype=jnp.float32)
        y_diff = jnp.zeros((half,), dtype=jnp.float32)

        X = jnp.concatenate([X_same, X_diff], axis=0)
        y = jnp.concatenate([y_same, y_diff], axis=0)

        # shuffle consistently
        k_shuffle = self.generate_subkey()
        perm = random.permutation(k_shuffle, self.num_samples)
        X = X[perm]
        y = y[perm]

        return X, y

    def generate_tf_dataset(self, batch_size, shuffle=True):
        """
        Create tf.data.Dataset from generated tensors.
        """
        X, y = self.generate_jax_tensors()
        ds = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            subkey = self.generate_subkey()
            ds = ds.shuffle(
                buffer_size=int(X.shape[0]),
                reshuffle_each_iteration=True,
                seed=int(subkey[0]),
            )

        ds = ds.batch(batch_size, drop_remainder=True).prefetch(2)
        return ds
    

class ContextIdentification:
    """
    Context identification task implemented as TensorFlow dataset wrapper.

    Each trial involves a block of stimuli where the zeroth stimuli is the reference
    stimuli and the first through the last stimuli are the context. The task is 
    to then output which item in the context that is most similar to the reference 
    stimuli.

    There are two options for the labe: feature-based or positional-based.
    - Feature-based :: the output is the item that is most similar
    - Positional-based :: the output is the idex of the most similar item
    In both of these, the feature dimension is the same as the number of items in the
    block.

    Uses JAX PRNG for reproducibility and exports a tf.data.Dataset.
    """
    def __init__(
        self,
        key,
        num_samples=1000,
        num_vectors_and_vector_len=10,
        allow_ref_in_context=True,
        label_mode="feature",
        max_resample_attempts=1024,
    ):
        """Initialize the ContextIdentification task.

        Args:
            key: JAX PRNGKey.
            num_samples: Number of trials to generate.
            num_vectors_and_vector_len: Both the block size and the one-hot feature dimension (L).
                Each example contains `L` vectors of length `L`: 1 reference + (L-1) context items.
            allow_ref_in_context: If False, the reference one-hot index is forbidden from appearing
                anywhere in the context.
            label_mode: Either "feature" or "position".
                - "feature": target is the one-hot feature vector of the closest context item.
                - "position": target is the integer index (0..L-2) of the closest context item.
            max_resample_attempts: Safety cap for rejection sampling.
        """
        assert num_vectors_and_vector_len >= 3, "Need at least 1 reference and 2 context items"
        assert label_mode in ["feature", "position"], "label_mode must be 'feature' or 'position'"

        self.key = key
        self.num_samples = int(num_samples)
        self.L = int(num_vectors_and_vector_len)
        self.num_vectors = self.L
        self.vector_len = self.L
        self.context_size = self.L - 1

        self.allow_ref_in_context = bool(allow_ref_in_context)
        self.label_mode = label_mode
        self.max_resample_attempts = int(max_resample_attempts)

    def generate_subkey(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def _one_hot(self, idx):
        return jnp.eye(self.vector_len, dtype=jnp.float32)[idx]

    def _sample_one_example(self, key):
        """Sample one example with rejection sampling to avoid distance ties.

        Returns:
            X: (L, L) float32 where X[0] is the reference and X[1:] is the context.
            y: if label_mode=='feature': (L,) float32 one-hot of closest context item.
               if label_mode=='position': () int32 index in 0..L-2 of closest context item.
        """
        L = self.L
        k_ref, k_ctx = random.split(key)

        ref_idx = random.randint(k_ref, (), 0, L)  # scalar

        # Rejection sampling: ensure unique closest context item (no argmin ties)
        def cond_fun(state):
            attempt, ok, _ctx_idxs = state
            return jnp.logical_and(jnp.logical_not(ok), attempt < self.max_resample_attempts)

        def body_fun(state):
            attempt, _ok, ctx_idxs = state
            # Use a different key each attempt
            k_try = random.fold_in(k_ctx, attempt)

            # Sample context indices WITH replacement (repeats allowed/likely)
            ctx_idxs = random.randint(k_try, (L - 1,), 0, L)

            # Optionally forbid reference index appearing in context
            if not self.allow_ref_in_context:
                # mark invalid if any equals ref
                has_ref = jnp.any(ctx_idxs == ref_idx)
            else:
                has_ref = False

            # distance in index space (no wrap-around)
            d = jnp.abs(ctx_idxs - ref_idx)  # (L-1,)
            min_d = jnp.min(d)
            # tie exists if more than one achieves min distance
            n_min = jnp.sum(d == min_d)
            tie = n_min > 1

            ok = jnp.logical_and(jnp.logical_not(has_ref), jnp.logical_not(tie))
            return (attempt + 1, ok, ctx_idxs)

        init_state = (jnp.array(0, dtype=jnp.int32), jnp.array(False), jnp.zeros((L - 1,), dtype=jnp.int32))
        attempt, ok, ctx_idxs = jax.lax.while_loop(cond_fun, body_fun, init_state)

        # In the rare case we hit max attempts, just proceed with the last sample
        d = jnp.abs(ctx_idxs - ref_idx)
        closest_pos = jnp.argmin(d).astype(jnp.int32)  # 0..L-2
        closest_idx = ctx_idxs[closest_pos]

        # Build X
        ref_vec = self._one_hot(ref_idx)
        ctx_vecs = jnp.stack([self._one_hot(i) for i in ctx_idxs], axis=0)
        X = jnp.concatenate([ref_vec[None, :], ctx_vecs], axis=0)  # (L, L)

        # Targets
        if self.label_mode == "feature":
            y = self._one_hot(closest_idx)  # (L,)
        else:
            y = closest_pos  # scalar int

        return X, y

    def generate_jax_tensors(self):
        """Generate full dataset as JAX arrays.

        Returns:
            X: (num_samples, L, L) float32
            y: if label_mode=='feature': (num_samples, L) float32
               if label_mode=='position': (num_samples,) int32
        """
        keys = random.split(self.generate_subkey(), self.num_samples)
        Xs, ys = [], []
        for k in keys:
            X, y = self._sample_one_example(k)
            Xs.append(X)
            ys.append(y)

        X = jnp.stack(Xs, axis=0)
        y = jnp.stack(ys, axis=0)

        # shuffle consistently
        k_shuffle = self.generate_subkey()
        perm = random.permutation(k_shuffle, self.num_samples)
        X = X[perm]
        y = y[perm]

        return X, y

    def generate_tf_dataset(self, batch_size, shuffle=True):
        """Create a tf.data.Dataset from generated tensors."""
        X, y = self.generate_jax_tensors()
        ds = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            subkey = self.generate_subkey()
            ds = ds.shuffle(
                buffer_size=int(X.shape[0]),
                reshuffle_each_iteration=True,
                seed=int(subkey[0]),
            )

        ds = ds.batch(batch_size, drop_remainder=True).prefetch(2)
        return ds
