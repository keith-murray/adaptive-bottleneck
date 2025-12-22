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