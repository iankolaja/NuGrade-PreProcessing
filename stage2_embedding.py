"""Stage 2: turn report PDFs into sentence embeddings and per-report similarity features.

    python stage2_embedding.py --pdf-dir pdfs --db output/nugrade_data.db

For each PDF named by its EXFOR entry, extracts text, gates it for quality, embeds each
surviving sentence with SciBERT, and stores:

  * `sentence_embeddings` — one row per sentence, used by the Flask app's RAG tools
  * `report_embeddings`   — one row per report: nine `{category}_max_sim` features plus a
                            mean embedding, used as KNN features by stage 3

Documents that fail the quality gate are skipped with a printed reason rather than embedded.
That matters beyond tidiness: every stored sentence is retrievable by the agent, and the
`{category}_max_sim` features are a *max* over sentences, so one mangled line that happens
to embed near a template inflates that report's score and corrupts its KNN neighbours.
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from embedding_backend import (
    EMBEDDING_DTYPE,
    embedder_from_config,
    extract_pdf_text,
    splitter_from_config,
)
from pipeline_config import Config, ConfigError, add_config_arguments, check_or_raise
from report_quality import document_quality_report, strip_reference_list
from stage_result import ProgressTracker, StageResult, null_printer, printer

# Similarity is penalised for sentences containing none of a category's keywords, so a
# lexically unrelated sentence cannot win on embedding proximity alone.
KEYWORD_GATE_WEIGHT = 0.9

PAGE_FURNITURE = re.compile(r"doi:|copyright|received:|accepted:", re.I)


def normalize_text(text):
    """Join hyphenated line breaks and collapse whitespace runs."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", "\n", text)
    return re.sub(r"[ \t]+", " ", text)


def remove_page_artifacts(text):
    """Drop bare page numbers, very short header/footer scraps, and journal furniture."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if len(stripped) < 5:
            continue
        if PAGE_FURNITURE.search(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def keyword_gate_weights(sentences, keywords, gate_weight=KEYWORD_GATE_WEIGHT):
    """1.0 for sentences containing a category keyword, ``gate_weight`` otherwise."""
    patterns = [re.compile(rf"\b{re.escape(k.lower())}\b") for k in keywords]
    return np.array([
        1.0 if any(p.search(s.lower()) for p in patterns) else gate_weight
        for s in sentences
    ])


def cosine_similarity_matrix(a, b):
    """Rows of ``a`` against rows of ``b``, without a scikit-learn dependency."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def max_category_similarity(sentences, sentence_vectors, template_vectors, keywords):
    """Best keyword-gated similarity between any sentence and any template.

    Returns (score, matching sentence) — the sentence is kept for debugging, since a
    surprising score is usually explained by seeing what matched.
    """
    weights = keyword_gate_weights(sentences, keywords)
    similarity = cosine_similarity_matrix(sentence_vectors, template_vectors)
    weighted = weights[:, None] * similarity
    best_row = int(np.argmax(weighted.max(axis=1)))
    return float(weighted.max()), sentences[best_row]


def load_templates(template_file, embedder):
    """Load the category templates and embed their sentences once."""
    with open(template_file) as handle:
        templates = json.load(handle)
    for category in templates:
        templates[category]["vectors"] = embedder.encode(templates[category]["sentences"])
    return templates


def existing_entries(db_path):
    """Entries already embedded, so a re-run can skip them."""
    if not Path(db_path).is_file():
        return set()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "report_embeddings" not in tables:
            return set()
        return {r[0] for r in con.execute(
            "SELECT DISTINCT EXFOR_Entry FROM report_embeddings")}
    finally:
        con.close()


def read_report(path, *, splitter, extract_text):
    """PDF to quality-gated sentences. Returns the report dict from report_quality."""
    text = extract_text(path)
    text = normalize_text(text)
    text = strip_reference_list(text)
    text = remove_page_artifacts(text)
    return document_quality_report(splitter.split(text))


def embed_report(entry, sentences, templates, embedder):
    """Compute the per-report feature row and the per-sentence rows for one report."""
    vectors = embedder.encode(sentences)

    features = {"EXFOR_Entry": entry}
    for category, template in templates.items():
        score, matched = max_category_similarity(
            sentences, vectors, template["vectors"], template["keywords"])
        features[f"{category}_max_sim"] = score
        features[f"{category}_match"] = matched
    features["mean_embedding"] = vectors.mean(axis=0).astype(EMBEDDING_DTYPE).tobytes()

    sentence_rows = pd.DataFrame({
        "Text": sentences,
        "Sentence_Number": np.arange(1, len(sentences) + 1),
        "Embedding": [v.astype(EMBEDDING_DTYPE).tobytes() for v in vectors],
        "EXFOR_Entry": entry,
    })
    return features, sentence_rows


def persist_report(db_path, features, sentence_rows):
    """Append one report's rows, replacing any previous version of that entry.

    Deleting just this entry — rather than `if_exists='replace'` — is what makes the loop
    resumable: an interrupted run keeps the reports it already finished.
    """
    con = sqlite3.connect(db_path)
    try:
        entry = features["EXFOR_Entry"]
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "report_embeddings" in tables:
            con.execute("DELETE FROM report_embeddings WHERE EXFOR_Entry = ?", (entry,))
        if "sentence_embeddings" in tables:
            con.execute("DELETE FROM sentence_embeddings WHERE EXFOR_Entry = ?", (entry,))
        pd.DataFrame([features]).to_sql("report_embeddings", con, if_exists="append",
                                        index=False)
        sentence_rows.to_sql("sentence_embeddings", con, if_exists="append", index=False)
        con.commit()
    finally:
        con.close()


def run(config, *, embedder=None, splitter=None, extract_text=None, progress=None):
    """Embed every report PDF. Returns a StageResult."""
    emit = progress or printer("2")
    started = time.monotonic()
    check_or_raise(config, "2")
    config.ensure_directories()

    embedder = embedder or embedder_from_config(config)
    splitter = splitter or splitter_from_config(config)
    extract_text = extract_text or extract_pdf_text

    pdfs = sorted(Path(config.pdf_dir).glob("*.pdf"))
    if config.limit:
        pdfs = pdfs[:config.limit]

    already = existing_entries(config.db_path) if config.resume else set()
    if already:
        emit(f"resuming: {len(already)} reports already embedded")

    emit("loading category templates")
    templates = load_templates(config.template_file, embedder)

    warnings = []
    counts = {"embedded": 0, "skipped_quality": 0, "skipped_existing": 0, "sentences": 0}
    tracker = ProgressTracker(len(pdfs), emit, every=config.log_every, unit="reports")

    for path in pdfs:
        entry = path.stem
        if entry in already:
            counts["skipped_existing"] += 1
            tracker.advance()
            continue

        report = read_report(path, splitter=splitter, extract_text=extract_text)
        if not report["usable"]:
            reason = "; ".join(report["reasons"])
            warnings.append(f"{entry}: {reason}")
            emit(f"{entry} SKIPPED ({report['recommendation']}): {reason}")
            counts["skipped_quality"] += 1
            tracker.advance()
            continue

        features, sentence_rows = embed_report(
            entry, report["sentences"], templates, embedder)
        persist_report(config.db_path, features, sentence_rows)
        counts["embedded"] += 1
        counts["sentences"] += len(sentence_rows)
        tracker.advance(suffix=f"{entry}: {report['kept_sentences']}/"
                               f"{report['total_sentences']} sentences")

    tracker.finish()
    if counts["skipped_quality"]:
        emit(f"{counts['skipped_quality']} reports need re-OCR; try: "
             f"ocrmypdf --redo-ocr <entry>.pdf <entry>.pdf")

    return StageResult(
        stage="2",
        ok=True,
        counts=counts,
        warnings=warnings,
        outputs=[Path(config.db_path)],
        elapsed_s=time.monotonic() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args()

    config = Config.from_args(args)
    try:
        result = run(config, progress=null_printer if args.quiet else None)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    print(result.render())
    return result.exit_code()


if __name__ == "__main__":
    sys.exit(main())
