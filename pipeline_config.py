"""Configuration for the preprocessing pipeline.

The notebooks kept their paths as literals inside a cell, which meant changing them on the
cluster required editing a .ipynb. Everything tunable now lives here, resolved from three
sources in order of precedence:

    command-line flag  >  environment variable  >  default

Environment variables carry the cluster paths because they are the natural fit for a SLURM
submission script — see slurm_example.sh.

Validation is deliberately split in two. `validate_static` checks prerequisites that must
exist before anything runs; `validate_inputs` checks prerequisites another stage produces.
The split is what lets `--stages 3` fail in a second with "run stage 2 first" while
`--stages 1,2,3` does not trip over a table stage 2 is about to create.
"""
import argparse
import os
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

ENV_PREFIX = "NUGRADE_"

# Defaults for the cluster this project runs on. Override per-site with the environment
# variables named alongside each field.
DEFAULT_ENDF71 = "/global/scratch/users/co_nuclear/endf71x/<symbol>/<ZAID>.710nc"
DEFAULT_ENDF8 = "/global/home/groups/co_nuclear/serpent/xsdata/endf8/Lib80x/<symbol>/<ZAID>.800nc"

DEFAULT_REACTION_CHANNELS = {"N,TOT": 1, "N,EL": 2, "N,INL": 4, "N,G": 102}
DEFAULT_INNER_WEIGHTS = {"Energy_Logstd": 0.5, "Z_std": 1.5, "A_std": 1.0}


class ConfigError(RuntimeError):
    """A prerequisite is missing or a setting is unusable. Reported, never a traceback."""


@dataclass(frozen=True)
class Config:
    """Every path and tunable the pipeline needs.

    Frozen so a stage cannot quietly reconfigure itself mid-run; use `replace()` to derive
    a variant.
    """

    # --- stage 1 inputs (cluster) ---
    x4_db: Path = Path("sources/x4sqlite1.db")          # NUGRADE_X4_DB
    endf71_template: str = DEFAULT_ENDF71                # NUGRADE_ENDF71_TEMPLATE
    endf8_template: str = DEFAULT_ENDF8                  # NUGRADE_ENDF8_TEMPLATE
    allow_missing_evals: bool = False

    # --- stage 2 inputs ---
    pdf_dir: Path = Path("pdfs")                         # NUGRADE_PDF_DIR
    template_file: Path = Path("template_sentences.json")  # NUGRADE_TEMPLATE_SENTENCES
    scibert_model: str = "allenai/scibert_scivocab_uncased"  # NUGRADE_SCIBERT_MODEL
    spacy_model: str = "en_core_web_sm"                  # NUGRADE_SPACY_MODEL
    batch_size: int = 32                                 # NUGRADE_BATCH_SIZE

    # --- outputs ---
    output_dir: Path = Path("output")                    # NUGRADE_OUTPUT_DIR
    db_path: Path = Path("output/nugrade_data.db")       # NUGRADE_DB
    test_db_path: Path = Path("output/test_data.db")

    # --- stage 3 numerics ---
    k_neighbors: int = 5                                 # NUGRADE_K_NEIGHBORS
    inner_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_INNER_WEIGHTS))
    feature_weights: Mapping[str, float] | None = None   # None -> all 1.0, built by stage 3

    # --- channels ---
    reaction_channels: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_REACTION_CHANNELS))

    # --- run control ---
    resume: bool = True                                  # NUGRADE_RESUME
    limit: int | None = None
    log_every: int = 100                                 # NUGRADE_LOG_EVERY

    @classmethod
    def resolved(cls, *, output_dir=None, db_path=None, test_db_path=None, **kwargs):
        """Build a Config, deriving database paths from ``output_dir`` when not given.

        The notebooks hardcoded 'output/nugrade_data.db' separately from
        ``output_directory``, so pointing the run elsewhere silently only moved half the
        outputs. Deriving one from the other removes that failure mode.
        """
        output_dir = Path(output_dir) if output_dir is not None else cls.output_dir
        return cls(
            output_dir=output_dir,
            db_path=Path(db_path) if db_path else output_dir / "nugrade_data.db",
            test_db_path=Path(test_db_path) if test_db_path else output_dir / "test_data.db",
            **kwargs,
        )

    @classmethod
    def from_args(cls, args, env=None):
        """Resolve CLI flags over environment variables over defaults.

        ``env`` is a parameter rather than a direct ``os.environ`` read so precedence can be
        tested without mutating the real environment.
        """
        env = os.environ if env is None else env

        def pick(flag, env_name, default, cast=str):
            value = getattr(args, flag, None)
            if value is not None:
                return value if not isinstance(default, Path) else Path(value)
            raw = env.get(ENV_PREFIX + env_name)
            if raw is not None and raw != "":
                return _cast(raw, default, cast)
            return default

        defaults = cls()
        return cls.resolved(
            x4_db=pick("x4_db", "X4_DB", defaults.x4_db),
            endf71_template=pick("endf71", "ENDF71_TEMPLATE", defaults.endf71_template),
            endf8_template=pick("endf8", "ENDF8_TEMPLATE", defaults.endf8_template),
            allow_missing_evals=bool(getattr(args, "allow_missing_evals", False)),
            pdf_dir=pick("pdf_dir", "PDF_DIR", defaults.pdf_dir),
            template_file=pick("templates", "TEMPLATE_SENTENCES", defaults.template_file),
            scibert_model=pick("scibert_model", "SCIBERT_MODEL", defaults.scibert_model),
            spacy_model=pick("spacy_model", "SPACY_MODEL", defaults.spacy_model),
            batch_size=pick("batch_size", "BATCH_SIZE", defaults.batch_size, int),
            output_dir=pick("output_dir", "OUTPUT_DIR", defaults.output_dir),
            db_path=getattr(args, "db", None) or env.get(ENV_PREFIX + "DB") or None,
            k_neighbors=pick("k_neighbors", "K_NEIGHBORS", defaults.k_neighbors, int),
            resume=not getattr(args, "no_resume", False),
            limit=getattr(args, "limit", None),
            log_every=pick("log_every", "LOG_EVERY", defaults.log_every, int),
        )

    def eval_templates(self):
        """Return [(evaluation name, path template)] in the order stage 1 writes them."""
        return [("endf7-1", self.endf71_template), ("endf8", self.endf8_template)]

    def ensure_directories(self):
        """Create the output directories this run will write into.

        The notebooks assumed these existed; a fresh checkout or a redirected --output-dir
        then failed partway through a long run.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        for name, _ in self.eval_templates():
            (self.output_dir / "evals" / name).mkdir(parents=True, exist_ok=True)


def _cast(raw, default, cast):
    """Convert an environment string to the type implied by the default."""
    if isinstance(default, Path):
        return Path(raw)
    if isinstance(default, bool):
        return raw.strip().lower() not in ("0", "false", "no", "")
    if isinstance(default, int) and cast is not str:
        return int(raw)
    return cast(raw)


def template_root(template):
    """The fixed directory prefix of a path template, before the first placeholder.

    '/data/endf8/<symbol>/<ZAID>.800nc' -> Path('/data/endf8'). Used to check that the
    evaluation library is reachable without knowing which nuclides will be requested.
    """
    head = str(template).split("<")[0]
    return Path(head).parent if not head.endswith(("/", os.sep)) else Path(head)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_static(config, stage):
    """Prerequisites that must exist before the pipeline starts, whatever else runs.

    Returns a list of problem strings; empty means the stage can start. Each message names
    the flag *and* the environment variable, because roughly half of these runs are SLURM
    scripts where only the latter is in play.
    """
    problems = []

    if stage in ("1", "1b"):
        if stage == "1" and not Path(config.x4_db).is_file():
            problems.append(
                f"missing X4Pro database: {config.x4_db}\n"
                f"    set --x4-db PATH or {ENV_PREFIX}X4_DB "
                f"(download from https://nds.iaea.org/cdroms/)")
        if stage == "1" and not config.allow_missing_evals:
            for name, template in config.eval_templates():
                root = template_root(template)
                if not root.is_dir():
                    problems.append(
                        f"no {name} evaluation files under {root}\n"
                        f"    this stage needs cluster ACE files; set the matching "
                        f"--endf71/--endf8 flag or {ENV_PREFIX}ENDF71_TEMPLATE/"
                        f"{ENV_PREFIX}ENDF8_TEMPLATE,\n"
                        f"    or pass --allow-missing-evals to record measurements with no "
                        f"evaluation comparison")

    if stage == "2":
        pdf_dir = Path(config.pdf_dir)
        if not pdf_dir.is_dir():
            problems.append(
                f"missing PDF directory: {pdf_dir}\n"
                f"    set --pdf-dir PATH or {ENV_PREFIX}PDF_DIR")
        elif not any(pdf_dir.glob("*.pdf")):
            problems.append(
                f"no PDFs in {pdf_dir}\n"
                f"    reports are named by EXFOR entry, e.g. 10283.pdf")
        if not Path(config.template_file).is_file():
            problems.append(
                f"missing template sentences: {config.template_file}\n"
                f"    set --templates PATH or {ENV_PREFIX}TEMPLATE_SENTENCES")

    return problems


# What each stage needs another stage to have produced.
STAGE_REQUIRES = {
    "1": set(),
    "1b": {"measurements"},
    "2": set(),
    "3": {"measurements", "entries", "subentries", "report_embeddings"},
}


def validate_inputs(config, stage):
    """Prerequisites produced by an earlier stage, checked against the database."""
    required = STAGE_REQUIRES.get(stage, set())
    if not required:
        return []

    db_path = Path(config.db_path)
    if not db_path.is_file():
        return [f"missing database: {db_path}\n"
                f"    stage {stage} needs tables produced by an earlier stage; "
                f"run stage 1 (and stage 2) first"]

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        present = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()

    missing = sorted(required - present)
    if not missing:
        return []
    producer = "stage 2" if missing == ["report_embeddings"] else "stage 1"
    return [f"database {db_path} is missing table(s) {', '.join(missing)}\n"
            f"    these are produced by {producer}; run it first"]


def check_or_raise(config, stage, *, static=True, inputs=True):
    """Raise ConfigError listing every problem at once, rather than one per run."""
    problems = []
    if static:
        problems += validate_static(config, stage)
    if inputs:
        problems += validate_inputs(config, stage)
    if problems:
        raise ConfigError(f"stage {stage} cannot start:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Shared CLI surface
# ---------------------------------------------------------------------------

def add_config_arguments(parser):
    """Attach the shared flags, so every entry point exposes an identical surface.

    All defaults are None so that "flag absent" stays distinguishable from "flag set to the
    default value" — otherwise an environment variable could never take effect.
    """
    group = parser.add_argument_group("configuration (flag > environment > default)")
    group.add_argument("--x4-db", dest="x4_db", default=None,
                       help=f"X4Pro sqlite database [{ENV_PREFIX}X4_DB]")
    group.add_argument("--endf71", default=None,
                       help=f"ENDF/B-VII.1 ACE path template [{ENV_PREFIX}ENDF71_TEMPLATE]")
    group.add_argument("--endf8", default=None,
                       help=f"ENDF/B-VIII ACE path template [{ENV_PREFIX}ENDF8_TEMPLATE]")
    group.add_argument("--allow-missing-evals", action="store_true",
                       help="record measurements even when no evaluation files are found")
    group.add_argument("--pdf-dir", dest="pdf_dir", default=None,
                       help=f"directory of report PDFs [{ENV_PREFIX}PDF_DIR]")
    group.add_argument("--templates", default=None,
                       help=f"template_sentences.json [{ENV_PREFIX}TEMPLATE_SENTENCES]")
    group.add_argument("--scibert-model", dest="scibert_model", default=None,
                       help=f"HuggingFace model id [{ENV_PREFIX}SCIBERT_MODEL]")
    group.add_argument("--spacy-model", dest="spacy_model", default=None,
                       help=f"spaCy model name [{ENV_PREFIX}SPACY_MODEL]")
    group.add_argument("--output-dir", dest="output_dir", default=None,
                       help=f"output directory [{ENV_PREFIX}OUTPUT_DIR]")
    group.add_argument("--db", default=None,
                       help=f"database path [{ENV_PREFIX}DB; default <output-dir>/nugrade_data.db]")
    group.add_argument("--k-neighbors", dest="k_neighbors", type=int, default=None,
                       help=f"KNN neighbours [{ENV_PREFIX}K_NEIGHBORS]")
    group.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                       help=f"embedding batch size [{ENV_PREFIX}BATCH_SIZE]")
    group.add_argument("--limit", type=int, default=None,
                       help="process only the first N units (smoke runs)")
    group.add_argument("--no-resume", action="store_true",
                       help="recompute from scratch instead of skipping completed work")
    group.add_argument("--log-every", dest="log_every", type=int, default=None,
                       help=f"progress line cadence [{ENV_PREFIX}LOG_EVERY]")
    return parser
