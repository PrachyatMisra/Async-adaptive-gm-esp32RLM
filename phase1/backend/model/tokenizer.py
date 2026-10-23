"""
Phase 1 Tokenizer - DistilBERT with offline fallback
Paper Section IV.B: Reviews truncated/padded to 96 tokens using DistilBERT tokenizer
Advantages analyzed in Phase 1 documentation
"""
from typing import List, Dict, Optional
import re

class DistilBERTTokenizerWrapper:
    """
    DistilBERT tokenizer wrapper matching paper protocol.
    Falls back to deterministic whitespace tokenizer if transformers not available.
    """

    def __init__(self, max_len: int = 96, tokenizer_name: str = "distilbert-base-uncased"):
        self.max_len = max_len
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        self._use_fallback = False
        self._init_tokenizer()

    def _init_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                local_files_only=False,
                trust_remote_code=False
            )
            self._use_fallback = False
        except Exception as e:
            print(f"[tokenizer] HF tokenizer unavailable ({e.__class__.__name__}: {e}), using fallback")
            self._use_fallback = True
            self._init_fallback()

    def _init_fallback(self):
        """Simple deterministic fallback - not DistilBERT but preserves interface"""
        self.vocab = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102}
        # Build simple vocab from common words
        self.word_re = re.compile(r"[a-z0-9']+")

    def encode(self, text: str) -> Dict[str, List[int]]:
        """
        Encode text to 96 tokens, matching paper truncation/padding.
        Returns dict with input_ids and attention_mask
        """
        if not self._use_fallback and self._tokenizer is not None:
            try:
                encoded = self._tokenizer(
                    text,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors=None
                )
                return {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"]
                }
            except Exception:
                pass

        # Fallback tokenization
        return self._fallback_encode(text)

    def _fallback_encode(self, text: str) -> Dict[str, List[int]]:
        tokens = self.word_re.findall(text.lower())[:self.max_len-2]
        # Simple hash-based ids for fallback (deterministic)
        ids = [101]  # CLS
        for tok in tokens:
            # Deterministic hash to avoid OOV issues, range 1000-30000
            h = 0
            for c in tok:
                h = (h * 31 + ord(c)) & 0xFFFFFFFF
            ids.append(1000 + (h % 29000))
        ids.append(102)  # SEP
        # Pad
        pad_len = self.max_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [0] * pad_len
        return {"input_ids": ids, "attention_mask": attention_mask}

    def batch_encode(self, texts: List[str]) -> Dict[str, List[List[int]]]:
        input_ids = []
        attention_masks = []
        for t in texts:
            enc = self.encode(t)
            input_ids.append(enc["input_ids"])
            attention_masks.append(enc["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_masks}

    def decode(self, ids: List[int]) -> str:
        if not self._use_fallback and self._tokenizer is not None:
            try:
                return self._tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                pass
        return " ".join(str(i) for i in ids[:10]) + "..."

def analyze_tokenization_advantages():
    """
    Returns analysis of tokenization choices - for documentation/demo
    Based on paper vs Phase 2 comparison
    """
    return {
        "phase1_paper": {
            "tokenizer": "DistilBERT (WordPiece, 30522 vocab)",
            "seq_len": 96,
            "advantages": [
                "Subword handling: OOV words split into meaningful pieces",
                "Pretrained alignment: leverages DistilBERT pretraining knowledge",
                "Standard benchmark: comparable with literature",
                "Context preservation: 96 tokens captures most IMDB reviews"
            ],
            "disadvantages": [
                "Vocab storage: ~100KB+ vocabulary table",
                "Not suitable for 520KB SRAM microcontrollers",
                "Requires HuggingFace ecosystem"
            ],
            "paper_choice_rationale": "Controlled academic evaluation with standard tokenizer"
        },
        "phase2_esp32": {
            "tokenizer": "FNV-1a Hash (zero-vocab, 4096 buckets)",
            "seq_len": 32,
            "advantages": [
                "Zero vocab storage: 0 bytes for dictionary",
                "Constant memory: streaming-friendly for MCU",
                "Deterministic: identical C and Python implementations",
                "Collision rate: ~8 pairs with 4096 buckets vs 30 with 2047"
            ],
            "disadvantages": [
                "No subword semantics: hash collisions possible",
                "Not pretrained: random embedding initialization",
                "Domain shift: requires retraining for new domains"
            ],
            "phase2_rationale": "Essential for 520KB SRAM ESP32 deployment"
        },
        "comparison": {
            "memory": "Phase1: ~100KB vocab + 3.06M params vs Phase2: 0KB vocab + 510KB int8",
            "accuracy_tradeoff": "Phase1: F1 0.661 (1200 samples) vs Phase2: 92.33% (1600+ synthetic, not comparable)",
            "deployment": "Phase1 targets research understanding, Phase2 targets edge deployment"
        }
    }
