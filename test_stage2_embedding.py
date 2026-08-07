"""Tests for the report embedding stage.

Runs entirely without torch, transformers, spaCy or PyMuPDF: text extraction, sentence
splitting and embedding are all injected. Assertions are about orchestration — resume,
quality gating, persistence, the float32 contract — not about semantic quality, which
belongs to the model.
"""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from conftest import read_table, table_columns
from embedding_backend import HashEmbedder, RegexSplitter
from pipeline_config import Config, ConfigError
from stage_result import null_printer
import stage2_embedding

# Real methodology prose. Must exceed document_quality_report's 20-sentence floor, which
# exists to reject scans whose text layer failed — so a realistic fixture has to be a
# realistic length.
GOOD_REPORT = " ".join([
    "The neutron activation cross sections were measured using a sample of natural lithium.",
    "The efficiency of the detector was determined with a calibrated cobalt source.",
    "Corrections were applied for gamma ray self absorption within the sample material.",
    "The dead time of the counting system remained below one percent throughout.",
    "Background contributions were measured with the sample removed from the beam.",
    "The neutron flux was monitored continuously using a fission chamber downstream.",
    "Time of flight techniques resolved the incident neutron energy to within two percent.",
    "Statistical uncertainties were propagated through the full analysis chain.",
    "The sample was irradiated for a total of twenty four hours at constant power.",
    "Gamma ray spectra were recorded with a high purity germanium detector system.",
    "The energy resolution of the detector was measured at several reference energies.",
    "Coincidence counting was used to resolve the complex decay scheme of the product.",
    "The normalization was checked against a well known standard cross section.",
    "Scattering corrections were computed with a Monte Carlo transport calculation.",
    "The measured yields were corrected for decay during the counting interval.",
    "Neutron energies were determined from the flight path and the timing resolution.",
    "The uncertainty budget includes contributions from counting and normalization.",
    "Repeated measurements agreed within the quoted statistical uncertainties.",
    "The target thickness was measured by weighing before and after the irradiation.",
    "Systematic effects from beam intensity drift were monitored throughout the run.",
    "Comparison with previous measurements shows agreement within two standard deviations.",
    "The final results are tabulated together with their estimated uncertainties.",
])

# Verbatim OCR garbage from the corpus: the gate must reject this document.
BAD_REPORT = " ".join([
    "s. s. hasa~-, a. k. ci~aube¥ and m. l. sehgal physics department aligarh muslim.",
    "the (n, y) cross-sections at 24kev have been measured by many work- ers (1-~).",
    "(1) i. k. cflaubey and m. l. s]~hgal: nucl.",
    "phys., 66, 267 (1965). (~) a. k. chaubey and m. l. sxhgal: phys.",
])

TEMPLATES = {
    "detector_efficiency": {
        "sentences": ["The detector efficiency was calibrated with a standard source."],
        "keywords": ["efficiency", "detector"],
    },
    "dead_time": {
        "sentences": ["Dead time corrections were applied to the counting rates."],
        "keywords": ["dead time"],
    },
}


@pytest.fixture
def reports_dir(tmp_path):
    """Reports named as the real directory names them — by EXFOR entry, with a .pdf suffix.

    The files hold plain text rather than real PDF bytes; the injected ``extract_text``
    reads them directly. Keeping the suffix real means prerequisite validation is exercised
    exactly as it runs in production.
    """
    directory = tmp_path / "pdfs"
    directory.mkdir()
    (directory / "10001.pdf").write_text(GOOD_REPORT)
    (directory / "10002.pdf").write_text(GOOD_REPORT.replace("lithium", "beryllium"))
    (directory / "30077.pdf").write_text(BAD_REPORT)
    return directory


@pytest.fixture
def stage2_config(tmp_path, reports_dir):
    templates = tmp_path / "templates.json"
    templates.write_text(json.dumps(TEMPLATES))
    return Config.resolved(output_dir=tmp_path, pdf_dir=reports_dir,
                           template_file=templates, log_every=1000)


def run_stage2(config, **kwargs):
    return stage2_embedding.run(
        config,
        embedder=HashEmbedder(),
        splitter=RegexSplitter(),
        extract_text=lambda path: path.read_text(),
        progress=null_printer,
        **kwargs,
    )


class TestRun:
    def test_embeds_the_usable_reports(self, stage2_config):
        result = run_stage2(stage2_config)

        assert result.counts["embedded"] == 2

    def test_skips_the_ocr_damaged_report(self, stage2_config):
        """Entry 30077 is real corpus garbage; embedding it would poison RAG and KNN."""
        result = run_stage2(stage2_config)

        assert result.counts["skipped_quality"] == 1
        assert any("30077" in w for w in result.warnings)

    def test_writes_both_tables(self, stage2_config):
        run_stage2(stage2_config)

        reports = read_table(stage2_config.db_path, "report_embeddings")
        sentences = read_table(stage2_config.db_path, "sentence_embeddings")
        assert len(reports) == 2
        assert len(sentences) > 0

    def test_produces_a_similarity_feature_per_category(self, stage2_config):
        run_stage2(stage2_config)

        columns = table_columns(stage2_config.db_path, "report_embeddings")
        for category in TEMPLATES:
            assert f"{category}_max_sim" in columns

    def test_keeps_the_matching_sentence_for_debugging(self, stage2_config):
        """The notebook computed this and threw it away, leaving scores unexplainable."""
        run_stage2(stage2_config)

        columns = table_columns(stage2_config.db_path, "report_embeddings")
        assert "detector_efficiency_match" in columns

    def test_sentence_numbers_start_at_one_and_are_contiguous(self, stage2_config):
        run_stage2(stage2_config)

        sentences = read_table(stage2_config.db_path, "sentence_embeddings")
        for _, group in sentences.groupby("EXFOR_Entry"):
            numbers = sorted(group["Sentence_Number"])
            assert numbers == list(range(1, len(numbers) + 1))

    def test_limit_restricts_the_work(self, stage2_config):
        limited = Config.resolved(output_dir=stage2_config.output_dir,
                                  pdf_dir=stage2_config.pdf_dir,
                                  template_file=stage2_config.template_file, limit=1)

        result = run_stage2(limited)

        assert result.counts["embedded"] + result.counts["skipped_quality"] == 1


class TestEmbeddingContract:
    def test_stored_vectors_are_float32(self, stage2_config):
        """Stage 3 reads these with np.frombuffer(..., float32); any other dtype silently
        produces garbage distances rather than an error."""
        run_stage2(stage2_config)

        reports = read_table(stage2_config.db_path, "report_embeddings")
        blob = reports.iloc[0]["mean_embedding"]
        recovered = np.frombuffer(blob, dtype=np.float32)
        assert len(recovered) == HashEmbedder().dim

    def test_mean_embedding_is_the_mean_of_the_sentence_vectors(self, stage2_config):
        run_stage2(stage2_config)

        reports = read_table(stage2_config.db_path, "report_embeddings")
        sentences = read_table(stage2_config.db_path, "sentence_embeddings")
        entry = reports.iloc[0]["EXFOR_Entry"]

        stored_mean = np.frombuffer(
            reports[reports["EXFOR_Entry"] == entry].iloc[0]["mean_embedding"],
            dtype=np.float32)
        vectors = np.vstack([
            np.frombuffer(b, dtype=np.float32)
            for b in sentences[sentences["EXFOR_Entry"] == entry]["Embedding"]])

        assert stored_mean == pytest.approx(vectors.mean(axis=0), abs=1e-6)

    def test_round_trips_through_stage3s_loader(self, stage2_config):
        """The cross-stage contract, exercised through the actual consumer."""
        import stage3_imputation

        run_stage2(stage2_config)
        con = sqlite3.connect(stage2_config.db_path)
        pd.DataFrame([{"EXFOR_Entry": "10001", "Uncertainty_Complete": 0}]).to_sql(
            "entries", con, index=False)
        pd.DataFrame([{"EXFOR_Entry": "10001"}]).to_sql("subentries", con, index=False)
        pd.DataFrame([{"EXFOR_Entry": "10001", "Energy": 1.0, "Data": 1.0,
                       "dData": None, "dData_assumed": 0.1, "Z": 3, "A": 7, "MT": 1,
                       "Projectile": "n"}]).to_sql("measurements", con, index=False)
        con.commit()
        con.close()

        _, reports, _ = stage3_imputation.load_inputs(stage2_config.db_path)

        assert reports.iloc[0]["mean_embedding"].dtype == np.float32


class TestResume:
    def test_skips_reports_already_embedded(self, stage2_config):
        run_stage2(stage2_config)

        second = run_stage2(stage2_config)

        assert second.counts["skipped_existing"] == 2
        assert second.counts["embedded"] == 0

    def test_no_resume_recomputes_everything(self, stage2_config):
        run_stage2(stage2_config)
        fresh = Config.resolved(output_dir=stage2_config.output_dir,
                                pdf_dir=stage2_config.pdf_dir,
                                template_file=stage2_config.template_file, resume=False)

        second = run_stage2(fresh)

        assert second.counts["embedded"] == 2

    def test_recomputing_does_not_duplicate_rows(self, stage2_config):
        """Delete-then-append per entry, never if_exists='replace' inside the loop."""
        run_stage2(stage2_config)
        before = len(read_table(stage2_config.db_path, "sentence_embeddings"))
        fresh = Config.resolved(output_dir=stage2_config.output_dir,
                                pdf_dir=stage2_config.pdf_dir,
                                template_file=stage2_config.template_file, resume=False)

        run_stage2(fresh)

        assert len(read_table(stage2_config.db_path, "sentence_embeddings")) == before

    def test_an_interrupted_run_keeps_finished_reports(self, stage2_config):
        """The reason resume is per-report rather than all-or-nothing."""
        class Failing(HashEmbedder):
            calls = 0

            def encode(self, texts):
                Failing.calls += 1
                if Failing.calls > 3:      # templates, then one report, then fail
                    raise RuntimeError("simulated interruption")
                return super().encode(texts)

        with pytest.raises(RuntimeError):
            stage2_embedding.run(stage2_config, embedder=Failing(),
                                 splitter=RegexSplitter(),
                                 extract_text=lambda p: p.read_text(),
                                 progress=null_printer)

        assert len(read_table(stage2_config.db_path, "report_embeddings")) >= 1


class TestTextCleaning:
    def test_joins_hyphenated_line_breaks(self):
        assert "measurements" in stage2_embedding.normalize_text("measure-\nments")

    def test_drops_bare_page_numbers(self):
        text = "Real sentence content here.\n42\nMore real content follows here."
        assert "\n42\n" not in stage2_embedding.remove_page_artifacts(text)

    def test_drops_journal_furniture(self):
        text = "Received: 3 March 1969\nThe detector was calibrated with a source."
        assert "Received" not in stage2_embedding.remove_page_artifacts(text)


class TestKeywordGate:
    def test_penalises_sentences_without_a_keyword(self):
        weights = stage2_embedding.keyword_gate_weights(
            ["the detector efficiency was measured", "unrelated prose here"],
            ["efficiency"])

        assert weights[0] == 1.0
        assert weights[1] < 1.0

    def test_matches_whole_words_only(self):
        weights = stage2_embedding.keyword_gate_weights(["inefficiency abounds"],
                                                        ["efficiency"])

        assert weights[0] < 1.0


class TestPrerequisites:
    def test_missing_pdf_directory_raises(self, tmp_path):
        templates = tmp_path / "t.json"
        templates.write_text(json.dumps(TEMPLATES))
        config = Config.resolved(output_dir=tmp_path, pdf_dir=tmp_path / "nope",
                                 template_file=templates)

        with pytest.raises(ConfigError, match="PDF directory"):
            run_stage2(config)

    def test_missing_templates_raises(self, tmp_path, reports_dir):
        config = Config.resolved(output_dir=tmp_path, pdf_dir=reports_dir,
                                 template_file=tmp_path / "missing.json")

        with pytest.raises(ConfigError, match="template sentences"):
            run_stage2(config)


class TestSchemaMigration:
    """A database created by an earlier version lacks columns this one writes."""

    def _legacy_table(self, db_path):
        """report_embeddings as the original notebook created it: no {category}_match."""
        import sqlite3
        con = sqlite3.connect(db_path)
        pd.DataFrame([{
            "EXFOR_Entry": "99999",
            "detector_efficiency_max_sim": 0.5,
            "dead_time_max_sim": 0.5,
            "mean_embedding": b"\x00" * 64,
        }]).to_sql("report_embeddings", con, index=False)
        con.commit()
        con.close()

    def test_embeds_into_a_table_missing_the_new_columns(self, stage2_config):
        """to_sql(append) cannot add columns, so this used to fail outright with
        'table report_embeddings has no column named background_treatment_match'."""
        self._legacy_table(stage2_config.db_path)

        result = run_stage2(stage2_config)

        assert result.counts["embedded"] == 2

    def test_the_new_columns_are_added(self, stage2_config):
        self._legacy_table(stage2_config.db_path)

        run_stage2(stage2_config)

        columns = table_columns(stage2_config.db_path, "report_embeddings")
        assert "detector_efficiency_match" in columns

    def test_existing_rows_survive_the_migration(self, stage2_config):
        """Widening must not discard what is already there."""
        self._legacy_table(stage2_config.db_path)

        run_stage2(stage2_config)

        reports = read_table(stage2_config.db_path, "report_embeddings")
        assert "99999" in set(reports["EXFOR_Entry"])
        legacy = reports[reports["EXFOR_Entry"] == "99999"].iloc[0]
        assert legacy["detector_efficiency_max_sim"] == 0.5
        assert pd.isna(legacy["detector_efficiency_match"])

    def test_migration_is_idempotent(self, stage2_config):
        self._legacy_table(stage2_config.db_path)
        run_stage2(stage2_config)
        first = table_columns(stage2_config.db_path, "report_embeddings")

        fresh = Config.resolved(output_dir=stage2_config.output_dir,
                                pdf_dir=stage2_config.pdf_dir,
                                template_file=stage2_config.template_file, resume=False)
        run_stage2(fresh)

        assert table_columns(stage2_config.db_path, "report_embeddings") == first
