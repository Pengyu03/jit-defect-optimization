"""Audit and chronologically split the public ApacheJIT full dataset.

Only training observations determine imputation and standardization. The CSV's
``year`` field is audited but never used for splitting: author_date (UTC Unix
seconds) is the single time authority. No label balancing or resampling occurs.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
LOG_FEATURES = [name for name in FEATURES if name != "ent"]
UTC = timezone.utc


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), UTC).isoformat()


def parse_label(text: str) -> int:
    label = text.strip().lower()
    if label in ("true", "1"):
        return 1
    if label in ("false", "0"):
        return 0
    raise ValueError(f"Unrecognized buggy label: {text!r}")


def parse_feature(text: str) -> tuple[float, str | None]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return np.nan, "missing_or_nonnumeric"
    if not np.isfinite(value):
        return np.nan, "nonfinite"
    if value < 0:
        return np.nan, "negative"
    return value, None


def class_counts(y: np.ndarray) -> dict:
    positive = int(y.sum())
    return {"n": int(len(y)), "negative": int(len(y) - positive),
            "positive": positive, "positive_fraction": float(y.mean())}


def prepare(raw: Path, out: Path, validation_fraction: float = 0.20) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    with raw.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty dataset")
    required = set(FEATURES + ["commit_id", "project", "buggy", "author_date", "year"])
    if required - set(rows[0]):
        raise ValueError(f"Missing fields: {sorted(required - set(rows[0]))}")

    # Abort on conflicting duplicate labels rather than silently choose one.
    by_commit: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        parse_label(row["buggy"])
        by_commit[row["commit_id"]].append(row)
    duplicate_groups = [group for group in by_commit.values() if len(group) > 1]
    conflicting = [group[0]["commit_id"] for group in duplicate_groups
                   if len({parse_label(row["buggy"]) for row in group}) > 1]
    if conflicting:
        raise ValueError(f"Conflicting duplicate labels; inspect before proceeding: {conflicting[:20]}")
    # If a future data revision contains matching duplicate labels, retain its
    # earliest author-date occurrence and record the explicit policy/count.
    rows = [min(group, key=lambda row: (float(row["author_date"]), row["project"]))
            for group in by_commit.values()]
    rows.sort(key=lambda row: (float(row["author_date"]), row["commit_id"]))

    timestamps = np.asarray([float(row["author_date"]) for row in rows], dtype=np.float64)
    if not np.isfinite(timestamps).all():
        raise ValueError("Nonfinite author_date")
    actual_years = np.asarray([datetime.fromtimestamp(t, UTC).year for t in timestamps])
    reported_years = np.asarray([int(row["year"]) for row in rows])
    y = np.asarray([parse_label(row["buggy"]) for row in rows], dtype=np.int64)
    ids = np.asarray([row["commit_id"] for row in rows])
    projects = np.asarray([row["project"] for row in rows])
    values = np.empty((len(rows), len(FEATURES)), dtype=np.float64)
    invalid = {name: Counter() for name in FEATURES}
    for i, row in enumerate(rows):
        for j, name in enumerate(FEATURES):
            values[i, j], reason = parse_feature(row[name])
            if reason:
                invalid[name][reason] += 1

    test_start = datetime(2017, 1, 1, tzinfo=UTC).timestamp()
    test_end = datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    earlier = np.flatnonzero(timestamps < test_start)
    if len(earlier) < 2:
        raise ValueError("Insufficient pre-2017 rows")
    candidate = int(np.floor((1.0 - validation_fraction) * len(earlier)))
    candidate = min(max(candidate, 1), len(earlier) - 1)
    val_start = timestamps[earlier[candidate]]
    masks = {
        "train": timestamps < val_start,
        "val": (timestamps >= val_start) & (timestamps < test_start),
        "test": (timestamps >= test_start) & (timestamps < test_end),
    }
    included = masks["train"] | masks["val"] | masks["test"]
    if not included.all():
        raise ValueError("Rows outside pre-2020 study period; revise protocol explicitly")
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Empty split")
    if any(len(np.unique(y[mask])) != 2 for mask in masks.values()):
        raise ValueError("Every split must contain both classes")

    transformed = values.copy()
    log_cols = [FEATURES.index(name) for name in LOG_FEATURES]
    transformed[:, log_cols] = np.log1p(transformed[:, log_cols])
    medians = np.nanmedian(transformed[masks["train"]], axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("All training values missing for at least one feature")
    transformed = np.where(np.isnan(transformed), medians[None, :], transformed)
    means = transformed[masks["train"]].mean(axis=0)
    stds = transformed[masks["train"]].std(axis=0, ddof=0)
    constant = stds == 0
    scales = np.where(constant, 1.0, stds)
    standardized = (transformed - means[None, :]) / scales[None, :]
    if not np.isfinite(standardized).all():
        raise ValueError("Nonfinite processed features")

    split_sets = [set(ids[mask]) for mask in masks.values()]
    assert all(not split_sets[i].intersection(split_sets[j])
               for i in range(3) for j in range(i + 1, 3))
    assert timestamps[masks["train"]].max() < timestamps[masks["val"]].min()
    assert timestamps[masks["val"]].max() < timestamps[masks["test"]].min()
    assert np.allclose(standardized[masks["train"]].mean(axis=0), 0, atol=1e-10)

    split_info = {}
    for name, mask in masks.items():
        per_project = {}
        for project in sorted(set(projects[mask])):
            per_project[project] = class_counts(y[mask & (projects == project)])
        split_info[name] = {
            **class_counts(y[mask]), "date_min_utc": iso(timestamps[mask].min()),
            "date_max_utc": iso(timestamps[mask].max()),
            "utc_year_counts": dict(sorted(Counter(map(int, actual_years[mask])).items())),
            "project_counts": per_project,
        }
    mismatch = actual_years != reported_years
    metadata = {
        "raw_path": str(raw.resolve()),
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "raw_rows": sum(len(group) for group in by_commit.values()),
        "retained_rows": len(rows), "features": FEATURES,
        "excluded_predictors": ["commit_id", "project", "buggy", "fix", "year", "author_date"],
        "audit": {
            "duplicate_commit_groups": len(duplicate_groups),
            "removed_duplicate_rows": sum(len(group) - 1 for group in duplicate_groups),
            "duplicate_policy": "Abort conflicting labels; otherwise retain earliest author_date occurrence",
            "conflicting_duplicate_labels": len(conflicting),
            "invalid_feature_values": {name: dict(counts) for name, counts in invalid.items()},
            "reported_year_vs_utc_author_year_mismatches": int(mismatch.sum()),
            "year_vs_timestamp_cross_2017_disagreements": int(np.sum((reported_years >= 2017) != (timestamps >= test_start))),
            "raw_reported_year_counts": dict(sorted(Counter(map(int, reported_years)).items())),
            "utc_author_year_counts": dict(sorted(Counter(map(int, actual_years)).items())),
            "project_identifier_count": int(len(set(projects))),
            "author_year_before_2003_count": int(np.sum(actual_years < 2003)),
            "author_year_after_2019_count": int(np.sum(actual_years > 2019)),
            "excluded_out_of_period_rows": 0,
            "date_min_utc": iso(timestamps.min()), "date_max_utc": iso(timestamps.max()),
            "year_mismatch_examples": [
                {"commit_id": rows[i]["commit_id"], "reported_year": int(reported_years[i]),
                 "author_date_utc": iso(timestamps[i])}
                for i in np.flatnonzero(mismatch)[:10]
            ],
        },
        "split_protocol": {
            "time_authority": "author_date Unix seconds converted to UTC; CSV year is audit-only",
            "validation_fraction_of_pre2017_target": validation_fraction,
            "validation_fraction_of_pre2017_actual": float(masks["val"].sum() / len(earlier)),
            "validation_start_utc": iso(val_start),
            "test_start_inclusive_utc": iso(test_start), "test_end_exclusive_utc": iso(test_end),
            "same_timestamp_policy": "All rows at validation boundary enter validation",
            "sampling": "None; retain natural class frequencies",
            "id_overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
        },
        "preprocessing": {
            "log1p_features": LOG_FEATURES, "untransformed_features": ["ent"],
            "invalid_value_policy": "Negative/nonfinite/nonnumeric to missing; train median after log transform",
            "fit_scope": "training observations only", "median_after_transform": medians.tolist(),
            "mean_after_imputation": means.tolist(), "population_std": stds.tolist(),
            "standardization_scale": scales.tolist(),
            "constant_training_features": [FEATURES[j] for j in np.flatnonzero(constant)],
        },
        "splits": split_info,
        "training_positive_weight_n0_over_n1": split_info["train"]["negative"] / split_info["train"]["positive"],
    }
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"features": np.asarray(FEATURES)}
    for name, mask in masks.items():
        arrays.update({f"X_{name}": np.ascontiguousarray(standardized[mask]),
                       f"y_{name}": y[mask], f"{name}_ids": ids[mask],
                       f"{name}_timestamps": timestamps[mask], f"projects_{name}": projects[mask]})
    np.savez_compressed(out / "dataset.npz", **arrays)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "split_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["commit_id", "project", "split", "buggy", "author_date", "author_date_utc", "reported_year"])
        for name, mask in masks.items():
            for i in np.flatnonzero(mask):
                writer.writerow([ids[i], projects[i], name, int(y[i]), int(timestamps[i]), iso(timestamps[i]), int(reported_years[i])])
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "apachejit_total.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    metadata = prepare(args.raw, args.out, args.validation_fraction)
    print(json.dumps({"rows": metadata["retained_rows"], "features": FEATURES,
                      "splits": {name: {k: v for k, v in info.items() if k != "project_counts"}
                                 for name, info in metadata["splits"].items()},
                      "audit": metadata["audit"], "output": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
