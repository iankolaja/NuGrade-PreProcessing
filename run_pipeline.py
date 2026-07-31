"""Run the whole preprocessing pipeline, or any subset of it, without Jupyter.

    python run_pipeline.py                        # everything
    python run_pipeline.py --stages 1             # ingestion only (the usual cluster job)
    python run_pipeline.py --stages 3             # re-impute an existing database
    python run_pipeline.py --dry-run              # validate configuration and stop

Cluster paths come from environment variables so a SLURM script can set them without
editing anything — see slurm_example.sh.

Exit codes are meaningful, because a SLURM job that fails silently is worse than one that
fails loudly:

    0  every selected stage succeeded
    1  a stage ran and could not finish
    2  a prerequisite is missing, or the configuration is unusable
"""
import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable

import stage1_ingestion
import stage2_embedding
import stage3_imputation
from pipeline_config import (
    Config,
    ConfigError,
    add_config_arguments,
    validate_inputs,
    validate_static,
)
from stage_result import StageResult, format_duration, null_printer, printer


@dataclass
class Stage:
    name: str
    label: str
    run: Callable
    produces: set
    requires: set


STAGES = {
    "1": Stage("1", "ingestion (measurements)", stage1_ingestion.run_measurements,
               produces={"measurements"}, requires=set()),
    "1b": Stage("1b", "ingestion (subentries/entries)", stage1_ingestion.run_aggregates,
                produces={"subentries", "entries"}, requires={"measurements"}),
    "2": Stage("2", "report embedding", stage2_embedding.run,
               produces={"report_embeddings", "sentence_embeddings"}, requires=set()),
    "3": Stage("3", "knn imputation", stage3_imputation.run,
               produces=set(),
               requires={"measurements", "entries", "subentries", "report_embeddings"}),
}

# Fixed dependency order. User order is ignored: running 3 before 2 is a mistake, not a
# preference.
STAGE_ORDER = ["1", "1b", "2", "3"]


def parse_stages(text):
    """Expand a --stages selection. Naming stage 1 implies its aggregate phase."""
    if not text or text.strip().lower() == "all":
        return list(STAGE_ORDER)

    selected = set()
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        if token not in STAGES:
            raise ConfigError(
                f"unknown stage {token!r}; choose from {', '.join(STAGE_ORDER)} or 'all'")
        selected.add(token)
        if token == "1":
            selected.add("1b")     # the notebook's two phases are one logical stage
    return [name for name in STAGE_ORDER if name in selected]


def preflight(config, stage_names):
    """Validate everything before doing any expensive work.

    Input checks are skipped for a stage whose producer is in this same selection —
    otherwise `--stages 1,2,3` would fail on a table stage 2 is about to create.
    """
    produced_here = set()
    for name in stage_names:
        produced_here |= STAGES[name].produces

    problems = []
    for name in stage_names:
        problems += validate_static(config, name)
        if not (STAGES[name].requires & produced_here):
            problems += validate_inputs(config, name)
    return problems


def run_pipeline(config, stage_names, *, keep_going=False, progress_factory=printer):
    """Run the selected stages in dependency order. Returns (results, skipped)."""
    results = {}
    skipped = {}
    failed_products = set()

    for name in stage_names:
        stage = STAGES[name]
        blocked = stage.requires & failed_products
        if blocked:
            skipped[name] = f"depends on {', '.join(sorted(blocked))}"
            continue

        emit = progress_factory(name)
        emit(f"starting {stage.label}")
        try:
            result = stage.run(config, progress=emit)
        except ConfigError as error:
            result = StageResult(stage=name, ok=False, warnings=[str(error)])
        results[name] = result

        if not result.ok:
            failed_products |= stage.produces
            if not keep_going:
                break

    return results, skipped


def render_summary(results, skipped, elapsed):
    lines = ["", "=" * 66, "PIPELINE SUMMARY", "=" * 66]
    for name in STAGE_ORDER:
        if name in results:
            result = results[name]
            status = "ok" if result.ok else "FAILED"
            counts = "  ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                               for k, v in list(result.counts.items())[:4])
            lines.append(f"  {name:3s} {STAGES[name].label:32s} {status:7s} "
                         f"{format_duration(result.elapsed_s):>8}  {counts}")
        elif name in skipped:
            lines.append(f"  {name:3s} {STAGES[name].label:32s} skipped  "
                         f"({skipped[name]})")
    lines.append(f"\n  total {format_duration(elapsed)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--stages", default="all",
                        help="comma-separated subset, e.g. '1' or '2,3' (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate configuration and inputs, then stop")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a failure instead of stopping")
    parser.add_argument("--skip-contract-check", action="store_true",
                        help="do not fail when the finished database violates the app "
                             "contract (useful while an earlier stage is still pending)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args()

    config = Config.from_args(args)
    try:
        stage_names = parse_stages(args.stages)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2

    print(f"database:  {config.db_path}")
    print(f"stages:    {', '.join(stage_names)}")

    problems = preflight(config, stage_names)
    if problems:
        print("\ncannot start:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\ndry run: configuration and inputs are valid; nothing was executed")
        for name in stage_names:
            print(f"  would run {name}: {STAGES[name].label}")
        return 0

    started = time.monotonic()
    factory = (lambda name: null_printer) if args.quiet else printer
    results, skipped = run_pipeline(config, stage_names,
                                    keep_going=args.keep_going,
                                    progress_factory=factory)
    print(render_summary(results, skipped, time.monotonic() - started))

    if any(not r.ok for r in results.values()):
        return 1
    if "3" in results and not args.skip_contract_check:
        if not _report_contract(config):
            # Every stage did its own job, but the database the app will actually load is
            # invalid. Reporting success here would let a SLURM job quietly ship it.
            return 1
    return 0


def _report_contract(config):
    """Check the finished database against the contract the Flask app enforces.

    Returns True when the database is usable. A failure here does not mean a stage
    misbehaved — it usually means an earlier stage has not been re-run yet — but it does
    mean the output is not shippable, so the caller turns it into a non-zero exit.
    """
    try:
        from validate_output_db import validate
    except ImportError:
        return True
    problems, warnings = validate(config.db_path)
    for warning in warnings:
        print(f"contract warning: {warning}")
    if problems:
        print("\ncontract check FAILED — the app will reject this database:")
        for problem in problems:
            print(f"  - {problem}")
        print("\n  if these are inherited from an earlier stage, re-run it "
              "(or use repair_derived_columns.py);\n"
              "  pass --skip-contract-check to accept the database anyway")
        return False
    print("\ncontract check passed")
    return True


if __name__ == "__main__":
    sys.exit(main())
