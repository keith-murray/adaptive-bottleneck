import jax
from jax import random
import jax.numpy as jnp
import jax_dataloader as jdl


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

    def generate_dataset(self, batch_size, shuffle=True):
        """Create a jdl.DataLoader from generated tensors."""
        X, y = self.generate_jax_tensors()
        ds = jdl.ArrayDataset(X, y)
        dataloader = jdl.DataLoader(
            ds, 
            'jax', 
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,
            generator=self.generate_subkey(),
        )
        return dataloader
    

class ContextIdentification:
    """
    Context identification task implemented as TensorFlow dataset wrapper.

    Each trial involves a block of stimuli where the zeroth stimuli is the reference
    stimuli and the first through the last stimuli are the context. The task is 
    to then output which item in the context that is most similar to the reference 
    stimuli.

    There are two label modes (both return integer labels for cross-entropy over L classes):
    - Feature mode: predict the *feature index* (0..L-1) of the closest context item.
    - Position mode: predict the *position in the block* X (0..L-1) of the closest context item.
      Since X[0] is always the reference, the label is never 0.

    Uses JAX PRNG for reproducibility and exports a tf.data.Dataset.
    """
    def __init__(
        self,
        key,
        num_samples=1000,
        num_vectors_and_vector_len=10,
        allow_ref_in_context=True,
        input_mode="self",
        label_mode="feature",
        geom_p=0.5,
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
        assert num_vectors_and_vector_len >= 4, (
            "Need L>=4 to guarantee a unique closest target while filling the rest of the context "
            "with strictly farther items (no ties)."
        )
        assert input_mode in ["self", "cross"]
        assert label_mode in ["feature", "position"], "label_mode must be 'feature' or 'position'"

        self.key = key
        self.num_samples = int(num_samples)
        self.L = int(num_vectors_and_vector_len)
        self.num_vectors = self.L
        self.vector_len = self.L
        self.context_size = self.L - 1

        self.allow_ref_in_context = bool(allow_ref_in_context)
        self.input_mode = input_mode
        self.label_mode = label_mode
        self.geom_p = float(geom_p)
        self.max_resample_attempts = int(max_resample_attempts)

    def generate_subkey(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def _one_hot(self, idx):
        return jnp.eye(self.vector_len, dtype=jnp.float32)[idx]

    def _sample_one_example(self, key):
        """Sample one example constructively to avoid distance ties.

        Returns:
            X: (L, L) float32 where X[0] is the reference and X[1:] is the context.
            y: integer label used for cross-entropy over L classes.
               - if label_mode=='feature': y is the *feature index* (0..L-1) of the closest context item.
               - if label_mode=='position': y is the *position in X* (0..L-1) of the closest context item.
                 Note: since X[0] is always the reference, y will never be 0.
        """
        L = self.L
        k_ref, k_dist, k_sign, k_ctx = random.split(key, 4)

        # 1) Sample the reference index
        ref_idx = random.randint(k_ref, (), 0, L).astype(jnp.int32)  # scalar

        # Compute the maximum possible distance from ref to any endpoint
        max_d = jnp.maximum(ref_idx, (L - 1) - ref_idx).astype(jnp.int32)  # in [0, L-1]

        # We must choose a target distance d such that there exists at least one index strictly farther.
        # That requires d < max_d, and also d >= 1 (target cannot equal reference).
        # With L>=4 this is always feasible.
        max_target_d = jnp.maximum(max_d - 1, 1).astype(jnp.int32)

        # 2) Sample the target distance from a truncated geometric distribution
        # jax.random.geometric(p) returns support {1,2,3,...}. We clip into [1, max_target_d].
        raw_d = random.geometric(k_dist, p=self.geom_p).astype(jnp.int32)
        d = jnp.minimum(raw_d, max_target_d).astype(jnp.int32)

        # Candidate targets are ref-d and ref+d, filtered to valid range [0, L-1]
        cand_minus = ref_idx - d
        cand_plus = ref_idx + d
        valid_minus = cand_minus >= 0
        valid_plus = cand_plus <= (L - 1)

        # Choose between valid candidates (no ties since distance is fixed)
        # If both are valid, sample one with a fair coin.
        choose_plus = random.bernoulli(k_sign, 0.5)
        target_idx = jnp.where(
            jnp.logical_and(valid_minus, valid_plus),
            jnp.where(choose_plus, cand_plus, cand_minus),
            jnp.where(valid_plus, cand_plus, cand_minus),
        ).astype(jnp.int32)

        # 3) Sample the rest of the context from the strictly-farther pool: |idx-ref| > d
        all_idxs = jnp.arange(L, dtype=jnp.int32)

        # Optionally forbid reference appearing in context (constructive pool already excludes distance 0 when d>=1,
        # but keep this for clarity / future modifications).
        not_ref = all_idxs != ref_idx

        farther = jnp.abs(all_idxs - ref_idx) > d
        pool_mask = jnp.logical_and(not_ref, farther)
        pool = all_idxs[pool_mask]

        # pool must be non-empty by construction when d < max_d
        pool_size = pool.shape[0]

        # Sample remaining L-2 context indices WITH replacement from pool
        # NOTE: repeats are allowed/likely because context_size == vector_len - 1.
        k_ctx_rest = random.fold_in(k_ctx, int(ref_idx))
        rest_pos = random.randint(k_ctx_rest, (L - 2,), 0, pool_size)
        rest_idxs = pool[rest_pos]

        # Assemble context indices (length L-1): first element is the unique closest target
        ctx_idxs = jnp.concatenate([jnp.array([target_idx], dtype=jnp.int32), rest_idxs], axis=0)

        # Shuffle context indices
        k_shuffle_ctx = random.fold_in(key, 999)
        perm = random.permutation(k_shuffle_ctx, ctx_idxs.shape[0])
        ctx_idxs = ctx_idxs[perm]

        # Recompute closest_pos as index in shuffled ctx_idxs where target_idx appears
        closest_pos = jnp.argmax(ctx_idxs == target_idx)

        # Build X: X[0] is reference, X[1:] is context
        ref_vec = self._one_hot(ref_idx)
        ctx_vecs = jnp.stack([self._one_hot(i) for i in ctx_idxs], axis=0)
        X = jnp.concatenate([ref_vec[None, :], ctx_vecs], axis=0)  # (L, L)

        # Targets (integer labels for CE over L classes)
        if self.label_mode == "feature":
            y = target_idx.astype(jnp.int32)           # feature index in 0..L-1
        elif self.input_mode == "self":
            y = (closest_pos + 1).astype(jnp.int32)     # position in X: 1..L-1 (0 is reference)
        else:
            y = (closest_pos).astype(jnp.int32)     # position in X: 0..L-2

        return X, y

    def generate_jax_tensors(self):
        """Generate full dataset as JAX arrays.

        Returns:
            X: (num_samples, L, L) float32
            y: (num_samples,) int32, integer labels for CE over L classes.
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

    def generate_dataset(self, batch_size, shuffle=True):
        """Create a jdl.DataLoader from generated tensors."""
        X, y = self.generate_jax_tensors()
        ds = jdl.ArrayDataset(X, y)
        dataloader = jdl.DataLoader(
            ds, 
            'jax', 
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,
            generator=self.generate_subkey(),
        )
        return dataloader
