"""Text extraction, sentence splitting and embedding, behind interfaces stage 2 can be
tested against.

Stage 2's real dependencies — PyMuPDF, spaCy, torch, transformers — are heavy, and SciBERT
requires a model download. Putting each behind a small protocol means the orchestration
(resume, quality gating, persistence) can be tested in milliseconds without any of them,
while production still uses the real thing.

The embedding contract is load-bearing and easy to break silently: `mean_embedding` is
stored as a raw buffer and read back with `np.frombuffer(..., dtype=np.float32)` in stage 3.
Every Embedder here must therefore return float32, and mean-pool over tokens to match how
the stored corpus was built.
"""
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

EMBEDDING_DTYPE = np.float32


class Embedder(Protocol):
    """Turns text into vectors. ``dim`` is the vector width."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dim) float32 array."""


class SentenceSplitter(Protocol):
    def split(self, text: str) -> list[str]:
        ...


@dataclass
class SciBertEmbedder:
    """Production embedder: SciBERT with attention-mask-weighted mean pooling.

    The tokenizer and model are instance attributes rather than module globals. In the
    notebook `get_embeddings_batch` read them from module scope, which made cell order
    load-bearing and the function impossible to call in isolation.

    Mean pooling — not the CLS token — is the corpus contract. The Flask app's query
    embedding must match it, or queries and documents end up in different vector spaces.
    """

    model_name: str = "allenai/scibert_scivocab_uncased"
    batch_size: int = 32
    max_length: int = 512
    dim: int = 768
    _tokenizer: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)
    _torch: object = field(default=None, repr=False)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=EMBEDDING_DTYPE)
        self._load()

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", truncation=True,
                                     max_length=self.max_length, padding=True)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            batches.append(pooled.numpy())
        return np.vstack(batches).astype(EMBEDDING_DTYPE)


@dataclass
class HashEmbedder:
    """Deterministic test embedder with no heavy dependencies.

    Vectors come from a hash of the text, so identical text embeds identically and the
    plumbing — shapes, dtype, persistence, round-tripping — is fully exercised. Similar
    sentences do *not* get similar vectors, so tests must never assert semantic ranking
    against this; that belongs to the model, not to this pipeline.
    """

    dim: int = 16

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=EMBEDDING_DTYPE)
        rows = []
        for text in texts:
            seed = abs(hash(text)) % (2 ** 32)
            rows.append(np.random.default_rng(seed).random(self.dim))
        return np.vstack(rows).astype(EMBEDDING_DTYPE)


@dataclass
class SpacySplitter:
    """Production sentence splitter.

    Two accommodations for the size of these documents. Only the sentence segmenter is
    loaded — the tagger, parser and NER are disabled — because they cost roughly 1 GB of
    working memory per 100,000 characters and contribute nothing to sentence boundaries.
    And text is processed in chunks, because spaCy refuses input over ``max_length``
    outright: ANL-7710 is 1.5 M characters and raised ValueError, which would have taken
    stage 2 down on the largest reports in the corpus.
    """

    model_name: str = "en_core_web_sm"
    chunk_size: int = 400_000
    _nlp: object = field(default=None, repr=False)

    def _load(self):
        if self._nlp is not None:
            return
        import spacy

        # senter is a lightweight statistical segmenter; the parser would also segment but
        # brings the memory cost that forces max_length in the first place.
        self._nlp = spacy.load(self.model_name, exclude=["ner", "lemmatizer"])
        if "senter" in self._nlp.pipe_names:
            self._nlp.enable_pipe("senter")

    def _chunks(self, text):
        """Split on paragraph breaks near the chunk size, so no sentence is cut in half."""
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start + self.chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            yield text[start:end]
            start = end

    def split(self, text):
        self._load()
        sentences = []
        for chunk in self._chunks(text):
            sentences.extend(s.text.strip() for s in self._nlp(chunk).sents)
        return [s for s in sentences if s]


@dataclass
class RegexSplitter:
    """Test splitter: break on sentence-ending punctuation. No spaCy needed."""

    def split(self, text):
        import re

        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def extract_pdf_text(path):
    """Production text extraction. PyMuPDF is imported lazily so tests need not install it."""
    import fitz

    document = fitz.open(str(path))
    try:
        return "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()


def embedder_from_config(config):
    return SciBertEmbedder(model_name=config.scibert_model, batch_size=config.batch_size)


def splitter_from_config(config):
    return SpacySplitter(model_name=config.spacy_model)
