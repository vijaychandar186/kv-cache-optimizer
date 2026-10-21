"""Deterministic, exact-length input IDs; tokenizer setup stays outside timed regions."""

import random
from pathlib import Path


def prompt_ids(tokenizer, length: int, seed: int, corpus: Path | None = None) -> list[int]:
    if length < 1:
        raise ValueError("Input length must be positive")
    rng = random.Random(seed)
    if corpus:
        text = corpus.read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < length:
            raise ValueError(
                f"Corpus has {len(ids)} tokens; need at least {length}. "
                "Provide a larger corpus; the lab does not repeat it silently."
            )
        start = rng.randrange(len(ids) - length + 1)
        return ids[start : start + length]
    # Random IDs measure controlled cache/compute pressure. They are not a quality benchmark.
    excluded = set(tokenizer.all_special_ids)
    vocab = sorted(set(tokenizer.get_vocab().values()) - excluded)
    if not vocab:
        raise ValueError("Tokenizer has no non-special tokens")
    return rng.choices(vocab, k=length)
