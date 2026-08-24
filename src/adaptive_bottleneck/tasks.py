import itertools

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import jax_dataloader as jdl


class XOR:
    """
    XOR task implemented as jax_dataloader wrapper.

    Each example is a pair of binary values encoded as 2-dimensional one-hot vectors:
    - "false" = [1, 0]
    - "true"  = [0, 1]

    Label = 1.0 if the two inputs differ (XOR=True), 0.0 otherwise (XOR=False).

    Input shape: (num_samples, 2, 2)
    Label shape: (num_samples,) float32
    """

    def __init__(self, key, num_samples=1000):
        self.key = key
        self.num_samples = num_samples

    def generate_subkey(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def generate_jax_tensors(self):
        """
        Returns:
            X: (num_samples, 2, 2) float32 — each row is [x1, x2] as one-hot vectors
            y: (num_samples,) float32 — XOR label (0.0 or 1.0)
        """
        k1, k2, k_shuffle = random.split(self.generate_subkey(), 3)

        x1_bits = random.randint(k1, (self.num_samples,), 0, 2)  # (B,) in {0, 1}
        x2_bits = random.randint(k2, (self.num_samples,), 0, 2)  # (B,) in {0, 1}

        y = (x1_bits != x2_bits).astype(jnp.float32)

        x1 = jnp.eye(2, dtype=jnp.float32)[x1_bits]  # (B, 2)
        x2 = jnp.eye(2, dtype=jnp.float32)[x2_bits]  # (B, 2)
        X = jnp.stack([x1, x2], axis=1)               # (B, 2, 2)

        perm = random.permutation(k_shuffle, self.num_samples)
        return X[perm], y[perm]

    def generate_dataset(self, batch_size, shuffle=True):
        """Create a jdl.DataLoader from generated tensors."""
        X, y = self.generate_jax_tensors()
        ds = jdl.ArrayDataset(X, y)
        return jdl.DataLoader(
            ds,
            'jax',
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            generator=self.generate_subkey(),
        )


class BlockComparison:
    """
    Block comparison task implemented as jax_dataloader wrapper.

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
    Context identification task implemented as jax_dataloader wrapper.

    Each trial involves a block of stimuli where the zeroth stimuli is the reference
    stimuli and the first through the last stimuli are the context. The task is 
    to then output which item in the context that is most similar to the reference 
    stimuli.

    There are three label modes (all return integer labels for cross-entropy over L classes):
    - Feature mode: predict the *feature index* (0..L-1) of the closest context item.
    - Position mode: predict the *position in the block* X (0..L-1) of the closest context item.
      Since X[0] is always the reference, the label is never 0.
    - Both mode: return both feature and position labels as a tuple (y_feature, y_position).

    Uses JAX PRNG for reproducibility.
    """
    def __init__(
        self,
        key,
        num_samples=1000,
        num_vectors=10,
        vector_len=10,
        allow_ref_in_context=True,
        input_mode="self",
        label_mode="feature",
        geom_p=0.5,
        max_resample_attempts=1024,
        positional_codes=False,
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
            positional_codes: If True, concatenate one-hot positional codes onto context rows
                and zero-pad the query row. X shape becomes (num_vectors, vector_len + context_size).
        """
        assert input_mode in ["self", "cross"]
        assert label_mode in ["feature", "position", "both"], "label_mode must be 'feature', 'position', or 'both'"

        self.key = key
        self.num_samples = int(num_samples)
        self.num_vectors = int(num_vectors)
        self.vector_len = int(vector_len)
        self.context_size = self.num_vectors - 1

        self.allow_ref_in_context = bool(allow_ref_in_context)
        self.input_mode = input_mode
        self.label_mode = label_mode
        self.geom_p = float(geom_p)
        self.max_resample_attempts = int(max_resample_attempts)
        self.positional_codes = bool(positional_codes)

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
        D = self.vector_len
        k_ref, k_dist, k_sign, k_ctx = random.split(key, 4)

        # 1) Sample the reference index
        ref_idx = random.randint(k_ref, (), 0, D).astype(jnp.int32)  # scalar

        # Compute the maximum possible distance from ref to any endpoint
        max_d = jnp.maximum(ref_idx, (D - 1) - ref_idx).astype(jnp.int32)  # in [0, L-1]

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
        valid_plus = cand_plus <= (D - 1)

        # Choose between valid candidates (no ties since distance is fixed)
        # If both are valid, sample one with a fair coin.
        choose_plus = random.bernoulli(k_sign, 0.5)
        target_idx = jnp.where(
            jnp.logical_and(valid_minus, valid_plus),
            jnp.where(choose_plus, cand_plus, cand_minus),
            jnp.where(valid_plus, cand_plus, cand_minus),
        ).astype(jnp.int32)

        # 3) Sample the rest of the context from the strictly-farther pool: |idx-ref| > d
        all_idxs = jnp.arange(D, dtype=jnp.int32)

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
        rest_pos = random.randint(k_ctx_rest, (self.context_size-1,), 0, pool_size)
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
        y_feature = target_idx.astype(jnp.int32)  # feature index in 0..L-1
        if self.input_mode == "self":
            y_position = (closest_pos + 1).astype(jnp.int32)  # position in X: 1..L-1 (0 is reference)
        else:
            y_position = closest_pos.astype(jnp.int32)  # position in X: 0..L-2

        # Optionally concatenate one-hot positional codes
        if self.positional_codes:
            pos_codes = jnp.eye(self.context_size, dtype=jnp.float32)  # (context_size, context_size)
            ctx_with_pos = jnp.concatenate([ctx_vecs, pos_codes], axis=1)  # (context_size, D + context_size)
            query_padded = jnp.concatenate([ref_vec, jnp.zeros(self.context_size, dtype=jnp.float32)])[None, :]
            X = jnp.concatenate([query_padded, ctx_with_pos], axis=0)

        if self.label_mode == "feature":
            return X, y_feature
        elif self.label_mode == "position":
            return X, y_position
        else:  # both
            return X, y_feature, y_position

    def generate_jax_tensors(self):
        """Generate full dataset as JAX arrays.

        Returns:
            X: (num_samples, L, L) float32
            y: (num_samples,) int32, integer labels for CE over L classes.
               If label_mode='both', returns (X, y_feature, y_position) instead.
        """
        keys = random.split(self.generate_subkey(), self.num_samples)

        if self.label_mode == "both":
            Xs, y_features, y_positions = [], [], []
            for k in keys:
                X, y_f, y_p = self._sample_one_example(k)
                Xs.append(X)
                y_features.append(y_f)
                y_positions.append(y_p)

            X = jnp.stack(Xs, axis=0)
            y_feature = jnp.stack(y_features, axis=0)
            y_position = jnp.stack(y_positions, axis=0)

            # shuffle consistently
            k_shuffle = self.generate_subkey()
            perm = random.permutation(k_shuffle, self.num_samples)
            X = X[perm]
            y_feature = y_feature[perm]
            y_position = y_position[perm]

            return X, y_feature, y_position
        else:
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

    def generate_all_examples(self):
        """Exhaustively enumerate every valid trial, for full-batch gradient descent.

        A trial is valid iff no context item is at distance 0 from the reference
        and exactly one context item achieves the (necessarily >=1) minimum
        distance to the reference -- precisely the population reachable by
        _sample_one_example's constructive rejection sampling. See
        scripts/context_identification/pos_codes/full_batch_combinatorics.py for
        the closed-form count N(L) this method is guaranteed to produce exactly
        (with context_size == vector_len, i.e. num_vectors == vector_len + 1).

        Returns:
            X: (N(L), num_vectors, vector_len [+ context_size if positional_codes]) float32
            y: (N(L),) int32, or (X, y_feature, y_position) if label_mode == "both"
        """
        D = self.vector_len
        L = self.context_size
        eye_D = np.eye(D, dtype=np.float32)
        eye_ctx = np.eye(self.context_size, dtype=np.float32)
        zeros_ctx = np.zeros(self.context_size, dtype=np.float32)

        Xs, y_features, y_positions = [], [], []

        for ref_idx in range(D):
            dists = np.abs(np.arange(D) - ref_idx)
            max_d = int(dists.max())
            ref_vec = eye_D[ref_idx]

            for d in range(1, max_d + 1):
                target_values = np.nonzero(dists == d)[0]
                farther_values = np.nonzero(dists > d)[0]
                if len(farther_values) == 0:
                    continue

                for target_val in target_values:
                    for closest_pos in range(L):
                        for rest in itertools.product(farther_values, repeat=L - 1):
                            ctx_idxs = list(rest)
                            ctx_idxs.insert(closest_pos, int(target_val))
                            ctx_vecs = eye_D[ctx_idxs]  # (L, D)

                            if self.positional_codes:
                                ctx_with_pos = np.concatenate([ctx_vecs, eye_ctx], axis=1)
                                query_padded = np.concatenate([ref_vec, zeros_ctx])[None, :]
                                X = np.concatenate([query_padded, ctx_with_pos], axis=0)
                            else:
                                X = np.concatenate([ref_vec[None, :], ctx_vecs], axis=0)

                            y_feature = int(target_val)
                            if self.input_mode == "self":
                                y_position = closest_pos + 1
                            else:
                                y_position = closest_pos

                            Xs.append(X)
                            y_features.append(y_feature)
                            y_positions.append(y_position)

        X = jnp.asarray(np.stack(Xs, axis=0))
        y_feature = jnp.asarray(np.array(y_features, dtype=np.int32))
        y_position = jnp.asarray(np.array(y_positions, dtype=np.int32))

        if self.label_mode == "feature":
            return X, y_feature
        elif self.label_mode == "position":
            return X, y_position
        else:  # both
            return X, y_feature, y_position

    def generate_dataset(self, batch_size, shuffle=True):
        """Create a jdl.DataLoader from generated tensors."""
        if self.label_mode == "both":
            X, y_feature, y_position = self.generate_jax_tensors()
            ds = jdl.ArrayDataset(X, y_feature, y_position)
        else:
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
