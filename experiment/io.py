from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import pandas as pd

from .variants import DatasetVariant

VARIANT_PLACEHOLDER = "{variant}"


def infer_output_format(path: Path) -> Literal["csv", "jsonl"]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return "jsonl"
    if suf in (".csv", ".tsv"):
        return "csv"
    raise ValueError(
        f"Unsupported output suffix {path.suffix!r}; use .csv, .tsv, or .jsonl"
    )


class IncrementalInferenceWriter:
    """Append inference rows to a file and ``flush`` every ``flush_every`` rows."""

    def __init__(self, path: Path | str, *, flush_every: int = 5) -> None:
        self.path = Path(path).expanduser().resolve()
        self.flush_every = max(1, int(flush_every))
        self._fmt = infer_output_format(self.path)
        self._fp: Any = None
        self._csv: csv.DictWriter | None = None
        self._n = 0

    def __enter__(self) -> IncrementalInferenceWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8", newline="")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None
        self._csv = None

    def write_row(self, row: dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("IncrementalInferenceWriter must be used as a context manager")
        if self._fmt == "jsonl":
            self._fp.write(json.dumps(row, ensure_ascii=False, default=str))
            self._fp.write("\n")
        else:
            if self._csv is None:
                fieldnames = list(row.keys())
                self._csv = csv.DictWriter(
                    self._fp,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                self._csv.writeheader()
            assert self._csv is not None
            self._csv.writerow({k: row.get(k, "") for k in self._csv.fieldnames})
        self._n += 1
        if self._n % self.flush_every == 0:
            self._fp.flush()


def write_inference_rows(rows: list[dict[str, Any]], path: Path | str) -> Path:
    """Write inference rows to disk (format from file suffix: ``.csv`` / ``.tsv`` / ``.jsonl``)."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = infer_output_format(p)
    if fmt == "csv":
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8")
    else:
        with p.open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False, default=str))
                f.write("\n")
    return p


def format_output_path_for_variant(
    template: Path | str,
    variant: DatasetVariant,
) -> Path:
    """Replace ``{variant}`` in the path string with ``variant.value`` (e.g. ``memory_only``)."""
    s = str(template)
    if VARIANT_PLACEHOLDER in s:
        s = s.replace(VARIANT_PLACEHOLDER, variant.value)
    return Path(s).expanduser().resolve()


def default_variant_output_path(
    out_dir: Path | str,
    variant: DatasetVariant,
    *,
    suffix: str = ".csv",
) -> Path:
    """``<out_dir>/inferences_<variant>_<UTC timestamp><suffix>`` (directory created)."""
    d = Path(out_dir).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return d / f"inferences_{variant.value}_{ts}{suffix}"
