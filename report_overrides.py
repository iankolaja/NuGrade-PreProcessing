"""Per-report handling rules, so awkward documents can be dealt with individually.

Most reports need no special treatment. A few do, and the reasons are specific enough that
no global threshold can express them:

  * an image-only scan that needs OCR before it can be embedded at all
  * a compiled laboratory report where one EXFOR entry cites a single chapter, so embedding
    all 477 pages describes the compilation rather than that entry's experiment
  * a non-English report that SciBERT should not embed
  * a document whose text is fine but unusual enough to trip a gate calibrated on the rest

The rules live in ``data/report_overrides.json`` rather than in the database, because they
are curation rather than derived data: the database is rebuilt wholesale by stage 1, which
would erase them. Keeping them in the repository also means a change is reviewable — an
override is a claim about a document, and the reason matters as much as the effect.

What stage 2 *applied* is recorded in the database afterwards, so a surprising embedding can
be traced back to the rule responsible.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("data/report_overrides.json")

# Recognised keys. Anything else is a typo, and a silently ignored typo in a curation file
# is worse than a loud failure — the override would simply never apply.
KNOWN_KEYS = {
    "reason",            # required: why this document needs special handling
    "skip",              # exclude entirely
    "action",            # what a human should do about it, e.g. "ocr"
    "pages",             # "12-30" or "5" — extract only these 1-based pages
    "strip_references",  # force reference truncation on or off
    "language",          # non-English marker, for documents SciBERT should not embed
    "duplicate_of",      # another entry sharing this document
    "min_usable_sentences",
    "max_rejection_rate",
    "healthy_yield",
}

GATE_KEYS = {"min_usable_sentences", "max_rejection_rate", "healthy_yield"}


class OverrideError(ValueError):
    """The override file is malformed. Raised rather than ignored, so typos surface."""


@dataclass
class Override:
    """Handling rules for one report."""

    entry: str
    reason: str = ""
    skip: bool = False
    action: str = ""
    pages: str = ""
    strip_references: bool | None = None
    language: str = ""
    duplicate_of: str = ""
    gate: dict = field(default_factory=dict)

    def page_numbers(self):
        """1-based page numbers this override selects, or None for the whole document."""
        return parse_pages(self.pages) if self.pages else None

    def describe(self):
        parts = []
        if self.skip:
            parts.append(f"skipped ({self.action or 'excluded'})")
        if self.pages:
            parts.append(f"pages {self.pages}")
        if self.strip_references is not None:
            parts.append(f"strip_references={self.strip_references}")
        if self.language:
            parts.append(f"language={self.language}")
        if self.gate:
            parts.append(", ".join(f"{k}={v}" for k, v in sorted(self.gate.items())))
        return "; ".join(parts) or "no effect"


def parse_pages(spec):
    """Parse '12-30', '5', or '3,12-14' into a sorted list of 1-based page numbers."""
    pages = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            first, last = int(match.group(1)), int(match.group(2))
            if first > last:
                raise OverrideError(f"page range {part!r} runs backwards")
            pages.update(range(first, last + 1))
        elif part.isdigit():
            pages.add(int(part))
        else:
            raise OverrideError(f"cannot parse page specification {part!r}")
    if not pages:
        raise OverrideError("empty page specification")
    return sorted(pages)


def load_overrides(path=DEFAULT_PATH):
    """Read the override file. Returns {entry: Override}; missing file means no overrides."""
    path = Path(path)
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise OverrideError(f"{path} is not valid JSON: {error}") from error

    entries = raw.get("reports", raw)
    overrides = {}
    for entry, rules in entries.items():
        if entry.startswith("_"):       # documentation keys
            continue
        if not isinstance(rules, dict):
            raise OverrideError(f"{path}: entry {entry} must be an object")

        unknown = set(rules) - KNOWN_KEYS
        if unknown:
            raise OverrideError(
                f"{path}: entry {entry} has unrecognised key(s) {sorted(unknown)}; "
                f"known keys are {sorted(KNOWN_KEYS)}")
        if not rules.get("reason"):
            raise OverrideError(
                f"{path}: entry {entry} needs a 'reason' — an override without one cannot "
                "be reviewed or retired later")
        if rules.get("pages"):
            parse_pages(rules["pages"])     # validate now, not mid-run

        overrides[entry] = Override(
            entry=entry,
            reason=rules.get("reason", ""),
            skip=bool(rules.get("skip", False)),
            action=rules.get("action", ""),
            pages=rules.get("pages", ""),
            strip_references=rules.get("strip_references"),
            language=rules.get("language", ""),
            duplicate_of=rules.get("duplicate_of", ""),
            gate={k: rules[k] for k in GATE_KEYS if k in rules},
        )
    return overrides


def extract_text_for(path, override, extract_text, extract_pages=None):
    """Extract text, honouring a page selection if the override specifies one.

    A page range matters for compiled reports. ANL-7710 is a 477-page annual report cited
    by two different EXFOR entries at different chapters; embedding all of it for both gives
    them identical feature vectors, so the KNN cannot tell two different experiments apart.
    """
    pages = override.page_numbers() if override else None
    if not pages:
        return extract_text(path)
    reader = extract_pages or extract_pdf_pages
    return reader(path, pages)


def extract_pdf_pages(path, pages):
    """Text of the given 1-based pages. PyMuPDF is imported lazily, as elsewhere."""
    import fitz

    document = fitz.open(str(path))
    try:
        wanted = [n for n in pages if 1 <= n <= document.page_count]
        return "\n".join(document[n - 1].get_text("text") for n in wanted)
    finally:
        document.close()


def gate_settings(override, defaults=None):
    """Quality-gate keyword arguments for a document, with any override applied."""
    settings = dict(defaults or {})
    if override:
        settings.update(override.gate)
    return settings


OVERRIDES_TABLE = """
    CREATE TABLE IF NOT EXISTS report_overrides_applied (
        EXFOR_Entry TEXT PRIMARY KEY,
        reason TEXT,
        effect TEXT,
        applied_at TEXT
    )
"""


def record_applied(connection, applied):
    """Record which overrides stage 2 acted on, so an odd embedding can be explained."""
    from datetime import datetime, timezone

    connection.execute(OVERRIDES_TABLE)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    connection.executemany(
        "INSERT OR REPLACE INTO report_overrides_applied "
        "(EXFOR_Entry, reason, effect, applied_at) VALUES (?,?,?,?)",
        [(o.entry, o.reason, o.describe(), stamp) for o in applied])
    connection.commit()
