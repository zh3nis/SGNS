import argparse
import os
import numpy as np


def save_word2vec_text(path, vectors):
    vocab_size, dim = vectors.shape
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{vocab_size} {dim}\n")
        for i in range(vocab_size):
            word = f"token_{i}"
            vals = " ".join(f"{x:.8f}" for x in vectors[i])
            f.write(f"{word} {vals}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random embeddings from N(0, I) and save as word2vec text."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings/random_gaussian",
        help="Output path for the word2vec text file.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=71290,
        help="Number of vectors to generate.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=200,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    vectors = rng.standard_normal(size=(args.vocab_size, args.dim), dtype=np.float32)
    save_word2vec_text(args.output, vectors)

    print(f"Saved random embeddings to {args.output}")
    print(f"Shape: {args.vocab_size} x {args.dim}")


if __name__ == "__main__":
    main()
