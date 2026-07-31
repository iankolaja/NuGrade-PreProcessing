"""Tests for per-report handling rules."""
import json
import sqlite3

import pytest

from report_overrides import (
    Override,
    OverrideError,
    extract_text_for,
    gate_settings,
    load_overrides,
    parse_pages,
    record_applied,
)


def write(tmp_path, reports):
    path = tmp_path / "report_overrides.json"
    path.write_text(json.dumps({"reports": reports}))
    return path


class TestParsePages:
    def test_single_page(self):
        assert parse_pages("5") == [5]

    def test_range_is_inclusive(self):
        assert parse_pages("12-15") == [12, 13, 14, 15]

    def test_mixed_list(self):
        assert parse_pages("3,12-14") == [3, 12, 13, 14]

    def test_tolerates_spaces(self):
        assert parse_pages(" 3 , 12 - 14 ") == [3, 12, 13, 14]

    def test_backwards_range_is_an_error(self):
        with pytest.raises(OverrideError, match="backwards"):
            parse_pages("30-12")

    def test_nonsense_is_an_error(self):
        with pytest.raises(OverrideError, match="cannot parse"):
            parse_pages("twelve")


class TestLoadOverrides:
    def test_missing_file_means_no_overrides(self, tmp_path):
        assert load_overrides(tmp_path / "absent.json") == {}

    def test_loads_rules(self, tmp_path):
        path = write(tmp_path, {"10104": {"reason": "image-only scan", "skip": True,
                                          "action": "ocr"}})

        overrides = load_overrides(path)

        assert overrides["10104"].skip
        assert overrides["10104"].action == "ocr"

    def test_a_reason_is_required(self, tmp_path):
        """An override nobody can explain cannot be reviewed or retired later."""
        path = write(tmp_path, {"10104": {"skip": True}})

        with pytest.raises(OverrideError, match="needs a 'reason'"):
            load_overrides(path)

    def test_an_unknown_key_is_an_error(self, tmp_path):
        """A silently ignored typo would mean the override never applies at all."""
        path = write(tmp_path, {"10104": {"reason": "x", "skipp": True}})

        with pytest.raises(OverrideError, match="unrecognised key"):
            load_overrides(path)

    def test_page_specification_is_validated_at_load(self, tmp_path):
        """Better to fail on load than three hours into a run."""
        path = write(tmp_path, {"10104": {"reason": "x", "pages": "30-12"}})

        with pytest.raises(OverrideError, match="backwards"):
            load_overrides(path)

    def test_documentation_keys_are_ignored(self, tmp_path):
        path = tmp_path / "o.json"
        path.write_text(json.dumps({"_about": "notes", "_keys": {},
                                    "reports": {"1": {"reason": "r"}}}))

        assert set(load_overrides(path)) == {"1"}

    def test_malformed_json_names_the_file(self, tmp_path):
        path = tmp_path / "o.json"
        path.write_text("{not json")

        with pytest.raises(OverrideError, match="not valid JSON"):
            load_overrides(path)

    def test_the_shipped_file_is_valid(self):
        """The real curation file must always load, or stage 2 cannot start."""
        overrides = load_overrides("data/report_overrides.json")

        assert overrides
        assert all(o.reason for o in overrides.values())


class TestExtractTextFor:
    def test_no_override_reads_the_whole_document(self, tmp_path):
        path = tmp_path / "x.pdf"

        text = extract_text_for(path, None, lambda p: "whole document")

        assert text == "whole document"

    def test_page_range_selects_pages(self, tmp_path):
        """A compiled report cited at one chapter should not embed all 477 pages: doing so
        describes the compilation, and gives two entries citing different chapters the same
        feature vector."""
        override = Override(entry="10374", reason="chapter", pages="2-3")
        captured = {}

        def reader(path, pages):
            captured["pages"] = pages
            return "chapter text"

        text = extract_text_for(tmp_path / "x.pdf", override,
                                lambda p: "whole", extract_pages=reader)

        assert text == "chapter text"
        assert captured["pages"] == [2, 3]

    def test_override_without_pages_reads_everything(self, tmp_path):
        override = Override(entry="1", reason="duplicate note")

        text = extract_text_for(tmp_path / "x.pdf", override, lambda p: "whole")

        assert text == "whole"


class TestGateSettings:
    def test_no_override_leaves_defaults(self):
        assert gate_settings(None, {"min_usable_sentences": 20}) == {
            "min_usable_sentences": 20}

    def test_override_replaces_a_threshold(self):
        override = Override(entry="1", reason="r", gate={"min_usable_sentences": 5})

        settings = gate_settings(override, {"min_usable_sentences": 20})

        assert settings["min_usable_sentences"] == 5

    def test_only_named_thresholds_are_touched(self):
        override = Override(entry="1", reason="r", gate={"healthy_yield": 50})

        settings = gate_settings(override, {"min_usable_sentences": 20})

        assert settings == {"min_usable_sentences": 20, "healthy_yield": 50}


class TestRecordApplied:
    def test_records_what_was_applied_and_why(self, tmp_path):
        """A surprising embedding should be traceable to the rule responsible."""
        con = sqlite3.connect(tmp_path / "db.sqlite")
        override = Override(entry="10104", reason="image-only scan", skip=True,
                            action="ocr")

        record_applied(con, [override])

        row = con.execute("SELECT EXFOR_Entry, reason, effect FROM "
                          "report_overrides_applied").fetchone()
        con.close()
        assert row[0] == "10104"
        assert row[1] == "image-only scan"
        assert "skipped" in row[2]

    def test_rerunning_replaces_rather_than_duplicates(self, tmp_path):
        con = sqlite3.connect(tmp_path / "db.sqlite")
        override = Override(entry="1", reason="r")

        record_applied(con, [override])
        record_applied(con, [override])

        count = con.execute("SELECT COUNT(*) FROM report_overrides_applied").fetchone()[0]
        con.close()
        assert count == 1


class TestDescribe:
    def test_summarises_the_effect(self):
        override = Override(entry="1", reason="r", pages="12-30",
                            gate={"healthy_yield": 50})

        description = override.describe()

        assert "pages 12-30" in description
        assert "healthy_yield=50" in description

    def test_a_note_only_override_says_so(self):
        """Duplicate cross-references change nothing yet; they record a known issue."""
        assert Override(entry="1", reason="r").describe() == "no effect"
