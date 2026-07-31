"""What a pipeline stage returns, and how it reports progress.

A stage returns a StageResult rather than printing and exiting, so the same `run()` serves
the command line, the pipeline runner, and a notebook cell.

The progress printer exists because of one specific failure mode: SLURM block-buffers
redirected stdout, so a stage that prints without flushing looks hung for hours. Every line
here is flushed.
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageResult:
    """Outcome of one stage.

    ``ok=False`` means the stage ran and could not finish its work — the reason is in
    ``warnings``. A missing prerequisite raises ConfigError instead; that is a setup
    mistake, not a result.
    """

    stage: str
    ok: bool = True
    counts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    elapsed_s: float = 0.0

    def exit_code(self):
        return 0 if self.ok else 1

    def merge(self, other):
        """Combine two phases of one logical stage into a single result."""
        return StageResult(
            stage=f"{self.stage}+{other.stage}",
            ok=self.ok and other.ok,
            counts={**self.counts, **other.counts},
            warnings=self.warnings + other.warnings,
            outputs=self.outputs + other.outputs,
            elapsed_s=self.elapsed_s + other.elapsed_s,
        )

    def render(self):
        """Human-readable block for a terminal or a SLURM log."""
        lines = [
            "=" * 66,
            f"stage {self.stage}: {'ok' if self.ok else 'FAILED'} "
            f"({format_duration(self.elapsed_s)})",
            "=" * 66,
        ]
        for key, value in self.counts.items():
            printable = f"{value:,}" if isinstance(value, int) else value
            lines.append(f"  {key:32s} {printable:>12}")
        if self.outputs:
            lines.append("  wrote:")
            lines.extend(f"    {path}" for path in self.outputs)
        if self.warnings:
            lines.append(f"  warnings ({len(self.warnings)}):")
            lines.extend(f"    - {w}" for w in self.warnings[:20])
            if len(self.warnings) > 20:
                lines.append(f"    ... and {len(self.warnings) - 20} more")
        return "\n".join(lines)

    def to_frame(self):
        """One-row DataFrame, for `display(result.to_frame())` in a notebook."""
        import pandas as pd

        row = {"stage": self.stage, "ok": self.ok,
               "elapsed": format_duration(self.elapsed_s), **self.counts,
               "warnings": len(self.warnings)}
        return pd.DataFrame([row])


def format_duration(seconds):
    """Seconds as h:mm:ss, or m:ss under an hour."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def printer(stage, stream=None):
    """Return a progress callback that timestamps and flushes every line.

    Flushing is not optional: SLURM block-buffers redirected stdout, so an unflushed stage
    produces no output at all until it finishes or dies.
    """
    stream = stream or sys.stdout
    start = time.monotonic()

    def emit(message):
        print(f"[{stage} {format_duration(time.monotonic() - start)}] {message}",
              file=stream, flush=True)

    return emit


def null_printer(_message):
    """Discard progress. Used by tests that assert on results rather than output."""


class ProgressTracker:
    """Periodic progress with a rate and an ETA.

    Stage 1 runs for hours, so a bare counter is not enough to tell a slow run from a stuck
    one — the rate is what makes that visible.
    """

    def __init__(self, total, emit, *, every=100, unit="items"):
        self.total = total
        self.emit = emit
        self.every = max(1, every)
        self.unit = unit
        self.done = 0
        self.start = time.monotonic()

    def advance(self, n=1, suffix=""):
        self.done += n
        if self.done % self.every and self.done != self.total:
            return
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        share = f" ({100 * self.done / self.total:.1f}%)" if self.total else ""
        eta = ""
        if rate > 0 and self.total and self.done < self.total:
            eta = f" eta {format_duration((self.total - self.done) / rate)}"
        tail = f" {suffix}" if suffix else ""
        self.emit(f"{self.done}/{self.total or '?'} {self.unit}{share} "
                  f"{rate:.1f}/s{eta}{tail}")

    def finish(self, suffix=""):
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        tail = f" {suffix}" if suffix else ""
        self.emit(f"done: {self.done} {self.unit} in {format_duration(elapsed)} "
                  f"({rate:.1f}/s){tail}")
