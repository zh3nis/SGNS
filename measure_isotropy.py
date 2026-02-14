import argparse
import numpy as np


def load_word2vec_text(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError("Expected header format: '<vocab_size> <dim>'")

        vocab_size, dim = map(int, header)
        vectors = np.empty((vocab_size, dim), dtype=np.float32)

        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                raise ValueError(
                    f"Line {i + 2}: expected {dim + 1} fields, got {len(parts)}"
                )
            vectors[i] = np.asarray(parts[1:], dtype=np.float32)

        if i + 1 != vocab_size:
            raise ValueError(
                f"Header says {vocab_size} rows, file contains {i + 1} rows"
            )

    return vectors


def isotropy_metrics(x, directions=128, pair_samples=20000, seed=0):
    n, d = x.shape
    eps = 1e-12
    rng = np.random.default_rng(seed)

    mean_norm = float(np.linalg.norm(x.mean(axis=0)))

    cov = (x.T @ x) / n
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, eps)
    cov_condition = float(eigvals.max() / eigvals.min())
    top_eigvar_ratio = float(eigvals.max() / eigvals.sum())

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_unit = x / np.maximum(norms, eps)

    u = rng.normal(size=(directions, d)).astype(np.float32)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), eps)
    dots = u @ x_unit.T
    z = np.exp(np.clip(dots, -20, 20)).sum(axis=1)
    partition_isotropy = float(z.min() / z.max())

    idx1 = rng.integers(0, n, size=pair_samples)
    idx2 = rng.integers(0, n, size=pair_samples)
    cos = (x_unit[idx1] * x_unit[idx2]).sum(axis=1)
    mean_abs_cos = float(np.mean(np.abs(cos)))

    return {
        "mean_norm": mean_norm,
        "cov_condition": cov_condition,
        "top_eigvar_ratio": top_eigvar_ratio,
        "partition_isotropy": partition_isotropy,
        "mean_abs_cos": mean_abs_cos,
    }


def print_metrics(title, metrics):
    print(title)
    print(f"  ||mean||: {metrics['mean_norm']:.6f}")
    print(f"  covariance condition number: {metrics['cov_condition']:.3f}")
    print(f"  top eigvar ratio: {metrics['top_eigvar_ratio']:.4f}")
    print(
        "  partition-function isotropy I=min(Z)/max(Z): "
        f"{metrics['partition_isotropy']:.4f}"
    )
    print(f"  mean |cos| over random pairs: {metrics['mean_abs_cos']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure isotropy metrics for word2vec text embeddings."
    )
    parser.add_argument(
        "embeddings_path",
        help="Path to embeddings in word2vec text format.",
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=128,
        help="Number of random directions for partition-function isotropy.",
    )
    parser.add_argument(
        "--pair_samples",
        type=int,
        default=20000,
        help="Number of random vector pairs for cosine concentration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Also report metrics after mean-centering vectors.",
    )
    args = parser.parse_args()

    x = load_word2vec_text(args.embeddings_path)
    print(f"Loaded embeddings: {x.shape[0]} words x {x.shape[1]} dims")

    raw = isotropy_metrics(
        x,
        directions=args.directions,
        pair_samples=args.pair_samples,
        seed=args.seed,
    )
    print_metrics("RAW", raw)

    if args.center:
        centered = x - x.mean(axis=0, keepdims=True)
        centered_metrics = isotropy_metrics(
            centered,
            directions=args.directions,
            pair_samples=args.pair_samples,
            seed=args.seed,
        )
        print_metrics("CENTERED", centered_metrics)


if __name__ == "__main__":
    main()
