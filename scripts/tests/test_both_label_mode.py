"""Sanity check for the 'both' label mode in ContextIdentification task."""

from jax import random
from adaptive_bottleneck.tasks import ContextIdentification


def main():
    # Test 'both' label mode
    task = ContextIdentification(
        key=random.key(42),
        num_samples=10,
        num_vectors=5,
        vector_len=5,
        input_mode="cross",
        label_mode="both",
    )

    X, y_f, y_p = task.generate_jax_tensors()
    print(f"X shape: {X.shape}")
    print(f"y_feature shape: {y_f.shape}, values: {y_f}")
    print(f"y_position shape: {y_p.shape}, values: {y_p}")

    # Test dataloader
    dl = task.generate_dataset(batch_size=4)
    for batch in dl:
        X_batch, y_f_batch, y_p_batch = batch
        print(f"Batch X: {X_batch.shape}, y_f: {y_f_batch.shape}, y_p: {y_p_batch.shape}")
        break

    print("\nSanity check passed!")


if __name__ == "__main__":
    main()
