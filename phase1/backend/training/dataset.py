"""
Phase 1 Dataset - IMDB subset 1200/300/600
Paper Section IV.B: Balanced subset, 96 tokens, DistilBERT tokenizer
"""
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass

@dataclass
class Split:
    texts: List[str]
    labels: List[int]

@dataclass
class Corpus:
    train: Split
    val: Split
    test: Split

# Synthetic lexicon for offline fallback - mirrors paper's sentiment phenomena
POS_WORDS = ["masterpiece", "brilliant", "outstanding", "superb", "phenomenal", "excellent", "captivating", "wonderful", "amazing", "fantastic"]
NEG_WORDS = ["terrible", "awful", "dreadful", "boring", "disappointing", "horrible", "worst", "pathetic", "lame", "disaster"]
TEMPLATES_POS = [
    "the movie was {adj} and {adj2}",
    "a {adj} film with {adj2} performances",
    "i loved this {adj} masterpiece",
    "this film is {adj}, truly {adj2}",
    "an {adj} and {adj2} experience",
]
TEMPLATES_NEG = [
    "the movie was {adj} and {adj2}",
    "a {adj} film, {adj2} and boring",
    "i hated this {adj} disaster",
    "this film is {adj}, completely {adj2}",
    "an {adj} and {adj2} failure",
]

def build_synthetic_imdb(seed: int = 42, n_train: int = 1200, n_val: int = 300, n_test: int = 600) -> Corpus:
    """Deterministic synthetic IMDB matching paper sizes"""
    rng = random.Random(seed)

    def gen_samples(n, pos_ratio=0.5):
        texts, labels = [], []
        for _ in range(n):
            is_pos = rng.random() < pos_ratio
            if is_pos:
                adj = rng.choice(POS_WORDS)
                adj2 = rng.choice(POS_WORDS)
                tmpl = rng.choice(TEMPLATES_POS)
                # Add contrastive clause 30% of time (paper's recursion test)
                if rng.random() < 0.3:
                    neg = rng.choice(["slow", "boring", "weak"])
                    text = tmpl.format(adj=adj, adj2=adj2) + f" although it starts {neg}, the ending is brilliant"
                else:
                    text = tmpl.format(adj=adj, adj2=adj2)
                labels.append(1)
            else:
                adj = rng.choice(NEG_WORDS)
                adj2 = rng.choice(NEG_WORDS)
                tmpl = rng.choice(TEMPLATES_NEG)
                if rng.random() < 0.3:
                    pos = rng.choice(["good", "decent", "okay"])
                    text = tmpl.format(adj=adj, adj2=adj2) + f" despite {pos} cast, it fails completely"
                else:
                    text = tmpl.format(adj=adj, adj2=adj2)
                labels.append(0)
            texts.append(text)
        return texts, labels

    train_t, train_l = gen_samples(n_train)
    val_t, val_l = gen_samples(n_val)
    test_t, test_l = gen_samples(n_test)

    return Corpus(
        train=Split(train_t, train_l),
        val=Split(val_t, val_l),
        test=Split(test_t, test_l)
    )

def load_imdb_subset(seed: int = 42, n_train: int = 1200, n_val: int = 300, n_test: int = 600, source: str = "auto") -> Corpus:
    """
    Load IMDB subset per paper. Tries HF datasets, falls back to synthetic.
    Paper: 1200 train, 300 val, 600 test, balanced
    """
    if source in ("auto", "hf"):
        try:
            from datasets import load_dataset
            print(f"[dataset] Loading HF imdb...")
            ds = load_dataset("imdb")
            # Shuffle and select
            train_ds = ds["train"].shuffle(seed=seed).select(range(n_train + n_val))
            test_ds = ds["test"].shuffle(seed=seed).select(range(n_test))

            train_texts = train_ds["text"][:n_train]
            train_labels = train_ds["label"][:n_train]
            val_texts = train_ds["text"][n_train:]
            val_labels = train_ds["label"][n_train:]
            test_texts = test_ds["text"]
            test_labels = test_ds["label"]

            print(f"[dataset] HF loaded: {len(train_texts)}/{len(val_texts)}/{len(test_texts)}")
            return Corpus(
                train=Split(train_texts, train_labels),
                val=Split(val_texts, val_labels),
                test=Split(test_texts, test_labels)
            )
        except Exception as e:
            print(f"[dataset] HF unavailable ({e.__class__.__name__}: {e}), using synthetic")

    print(f"[dataset] Using synthetic corpus {n_train}/{n_val}/{n_test}")
    return build_synthetic_imdb(seed, n_train, n_val, n_test)

def get_dataset_stats(corpus: Corpus) -> Dict:
    def stats(split: Split):
        return {
            "samples": len(split.texts),
            "pos": sum(split.labels),
            "neg": len(split.labels) - sum(split.labels),
            "avg_len": sum(len(t.split()) for t in split.texts) / len(split.texts) if split.texts else 0
        }
    return {
        "train": stats(corpus.train),
        "val": stats(corpus.val),
        "test": stats(corpus.test),
        "total": len(corpus.train.texts) + len(corpus.val.texts) + len(corpus.test.texts),
        "paper_spec": "1200 train / 300 val / 600 test = 2100 total, balanced, 96 tokens"
    }
