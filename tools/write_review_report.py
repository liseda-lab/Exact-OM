#!/usr/bin/env python3
"""Write one concise RESULTS.md record for every review-response run."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    name: str
    work_package: str
    arm: str
    config_delta: str
    dump_expected: bool
    note: str = ""
    run_yaml: str = "omim-ordo.yaml"
    dataset_label: str = "fixed 300-source OMIM–ORDO subset"
    reference_path: str = "data/omim-ordo/test.tsv"


RUNS = (
    RunRecord(
        "e1_dump",
        "WP1 / WP2",
        "full baseline and channel dump",
        "`inference_params.channel_dump: true`",
        True,
        "This single run is reused as WP2's full arm.",
    ),
    RunRecord(
        "e2_uniform_weights",
        "WP2",
        "uniform weights",
        "`model.params.uniform_weights: true`",
        True,
    ),
    RunRecord("e2_no_hier", "WP2", "no hierarchy", "`ablate_channels: [hier]`", True),
    RunRecord("e2_no_sim", "WP2", "no similarity", "`ablate_channels: [sim]`", True),
    RunRecord("e2_no_diff", "WP2", "no difference", "`ablate_channels: [diff]`", True),
    RunRecord(
        "e2_no_attr",
        "WP2",
        "no attributes",
        "`ablate_channels: [attr]`",
        True,
        (
            "The first attempt exposed an fp16-to-fp32 indexed-assignment error before batch 1. "
            "Its pre-failure application log is preserved as `failed_attempt_1_dtype.log`; the "
            "session traceback identified the dtype mismatch. The dtype-safe rerun completed and "
            "is the result reported here."
        ),
    ),
    RunRecord(
        "e2_lex_only",
        "WP2",
        "lexical only",
        "`ablate_channels: [hier, sim, diff, attr]`",
        True,
    ),
    RunRecord("e4_gamma_1", "WP4", "gamma = 1", "`gamma: 1.0`, `tau: 0.5`", False),
    RunRecord(
        "e4_gamma_2",
        "WP4",
        "gamma = 2 / tau = 0.5",
        "`gamma: 2.0`, `tau: 0.5`",
        False,
        "This one physical run is reused as the tau=0.5 setting.",
    ),
    RunRecord("e4_gamma_3", "WP4", "gamma = 3", "`gamma: 3.0`, `tau: 0.5`", False),
    RunRecord("e4_gamma_4", "WP4", "gamma = 4", "`gamma: 4.0`, `tau: 0.5`", False),
    RunRecord("e4_tau_0.4", "WP4", "tau = 0.4", "`gamma: 2.0`, `tau: 0.4`", False),
    RunRecord(
        "e4_tau_0.6",
        "WP4",
        "tau = 0.6",
        "`gamma: 2.0`, `tau: 0.6`",
        False,
        "The log records four gated/invoked decision pairs across 30,404 scored pairs and "
        "confirms the hosted decision stage's first concurrent wave completed successfully.",
    ),
    RunRecord(
        "e6_fma_dump",
        "WP6",
        "SNOMED–FMA Body channel dump",
        "`config.yaml` byte-identical to E1; only the dataset/run YAML changes",
        True,
        (
            "This deliberately uses E1's `tau_LLM: 1.0`, not the published FMA config's 0.5. "
            "Its subset metrics are context only and are not comparable to the published "
            "full-task result."
        ),
        run_yaml="snomed-fma-body.yaml",
        dataset_label="fixed 300-source SNOMED–FMA Body subset",
        reference_path="data/snomed-fma.body/test.tsv",
    ),
)

WP2_ROWS = (
    ("full", "e1_dump"),
    ("uniform weights", "e2_uniform_weights"),
    ("no hierarchy", "e2_no_hier"),
    ("no similarity", "e2_no_sim"),
    ("no difference", "e2_no_diff"),
    ("no attributes", "e2_no_attr"),
    ("lexical only", "e2_lex_only"),
)

WP4_ROWS = (
    ("gamma", "1", "e4_gamma_1"),
    ("gamma", "2", "e4_gamma_2"),
    ("gamma", "3", "e4_gamma_3"),
    ("gamma", "4", "e4_gamma_4"),
    ("tau", "0.4", "e4_tau_0.4"),
    ("tau", "0.5", "e4_gamma_2"),
    ("tau", "0.6", "e4_tau_0.6"),
)

WP1_HEADLINES = """## WP1 headline analyses

1. Reliability: `q_lex=1` on all 30,404 scored pairs; hierarchy Spearman correlation is 0.0048
   (95% CI [-0.0185, 0.0270]); similarity and difference are wholly inactive; and attribute
   Spearman correlation is -0.0359 (95% CI [-0.0469, -0.0243]). The observed evidence is adverse
   to a general rising quality–reliability claim.
2. Same-family confounders: median `q_lex` is 1.0 for both 3,604 confounders and 205 scored correct
   pairs; Mann–Whitney two-sided p=1.0 and histogram overlap=1.0. This does not rebut the
   reviewer's counterexample.
3. Both inertness tests (`sigma < 1e-6` and `omega < 0.01`) yield 82.74% for hierarchy, 100% for
   similarity, 100% for difference, and 0% for attributes.
4. LLM behavior: 0/30,404 pairs were gated and 0/30,404 invoked the LLM, so there were no
   LLM-induced top-1 changes.
5. Sanity identities pass: maximum structural-weight sum error is 8.94e-08 and maximum
   lexical/structural mixture error is 9.88e-08.

Full tables and plots are in `../analysis/` from this run folder."""

WP4_MISLABEL = """The historical
`exp/test/Full_local_bioml/omim-ordo-val-gamma-2` directory is mislabelled: it does not set
`gamma` and changes `tau_LLM` instead. It must not be cited as a gamma result."""

WP5_HEADLINES = """## WP5 reliability re-analysis

The three preregistered active-only tests used 2,000 bootstrap resamples and seed 42. Hierarchy is
not supported by conditional calibration (`rho=1.0000`, 95% CI `[0.9758, 1.0000]`); attributes
are supported for conditional calibration (`rho=-0.8424`, CI `[-0.9152, -0.5273]`) while their
source AUC and pick correlations remain null. Lexical quality is constant, and similarity and
difference are fully inactive, so those channels are untestable on this dump.

Full methods, per-channel tables, tie policy, and deterministic-rerun evidence are in
[`analysis/WP5_RESULTS.md`](analysis/WP5_RESULTS.md)."""

WP6_HEADLINES = """## WP6 FMA headline analyses

1. Pair accounting: the run found zero exact matches, processed all 32,724 expected candidate-pair
   occurrences, and wrote all 32,724 rows from 300 sources to the 35-column dump.
2. Activity: hierarchy is fully active; similarity and difference are each active on 1,075/32,724
   rows (3.2851%); attributes are fully active.
3. Difference pivot: 17,960/32,724 rows (54.8833%) have both positive difference quality and
   `abs(s_diff-tau)<1e-6`. This is 17,960/19,035 (94.3525%) of positive-quality difference rows.
4. Reliability: hierarchy, similarity, and difference are not supported by the primary
   conditional-calibration test; attributes are supported; lexical quality is constant and
   untestable.
5. Validation: all numeric data are finite, the dump covers 300 sources, and the omega-sum and
   `S_pair` identities pass at `1e-5`.

The measured wall clock is 18.1 minutes versus the preregistered 4.5–5 hour estimate. Ranking
metrics are context only because the 300-source subset and `tau_LLM=1.0` differ from the published
full-task setting.

Full distributions, reliability tables, reproducibility evidence, and artifact links are in
[`../analysis/WP6_RESULTS.md`](../analysis/WP6_RESULTS.md)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("exp/test/review_response"),
    )
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        values = {row["Metric"]: float(row["Value"]) for row in rows}
    required = {"MRR", "Hits@1", "Hits@5", "Hits@10"}
    if set(values) != required:
        raise ValueError(f"{path}: expected {sorted(required)}, found {sorted(values)}")
    return values


def read_total_minutes(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"(?m)^Total:\s*([0-9.]+)\s+minutes$", text)
    if not matches:
        raise ValueError(f"{path}: no Total timing")
    return float(matches[-1])


def render_wp2_table(runs_root: Path) -> str:
    baseline = read_metrics(runs_root / "e1_dump" / "evaluation_results.csv")["MRR"]
    rows = [
        "| Arm | Run | MRR | Hits@1 | Hits@5 | Hits@10 | ΔMRR | Wall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, name in WP2_ROWS:
        run_dir = runs_root / name
        metrics = read_metrics(run_dir / "evaluation_results.csv")
        wall = read_total_minutes(run_dir / "times.txt")
        rows.append(
            f"| {arm} | `{name}` | {metrics['MRR']:.3f} | {metrics['Hits@1']:.3f} | "
            f"{metrics['Hits@5']:.3f} | {metrics['Hits@10']:.3f} | "
            f"{metrics['MRR'] - baseline:+.3f} | {wall:.1f} min |"
        )
    return "\n".join(rows)


def render_wp4_table(runs_root: Path) -> str:
    rows = [
        "| Series | Setting | Physical run | MRR | Hits@1 | Hits@5 | Hits@10 | Wall |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for series, setting, name in WP4_ROWS:
        run_dir = runs_root / name
        metrics = read_metrics(run_dir / "evaluation_results.csv")
        wall = read_total_minutes(run_dir / "times.txt")
        reuse = " (reused)" if series == "tau" and setting == "0.5" else ""
        rows.append(
            f"| {series} | {setting} | `{name}`{reuse} | {metrics['MRR']:.3f} | "
            f"{metrics['Hits@1']:.3f} | {metrics['Hits@5']:.3f} | "
            f"{metrics['Hits@10']:.3f} | {wall:.1f} min |"
        )
    return "\n".join(rows)


def render(
    record: RunRecord,
    run_dir: Path,
    wp2_table: str,
    wp4_table: str,
) -> str:
    metrics = read_metrics(run_dir / "evaluation_results.csv")
    wall_clock = read_total_minutes(run_dir / "times.txt")
    artifacts = [
        "`config.yaml`",
        f"`{record.run_yaml}`",
        "`evaluation_results.csv`",
        "`exact.log`",
        "`times.txt`",
        "`model/alignment/src2tgt.maps_local.tsv`",
    ]
    if record.dump_expected:
        artifacts.append("`channel_dump.csv`")
    note = f"\n{record.note}\n" if record.note else "\n"
    if record.name == "e1_dump":
        findings = f"\n{WP1_HEADLINES}\n\n" "## Complete WP2 comparison\n\n" f"{wp2_table}\n"
    elif record.work_package == "WP2":
        uniform_note = ""
        if record.name == "e2_uniform_weights":
            uniform_note = (
                "\nThe configured `|K|` is 6: three hierarchy-family keys plus similarity, "
                "difference, and attributes. Each key receives 1/6; grouped dump weights are "
                "`omega_hier=0.5` and `omega_sim=omega_diff=omega_attr=1/6`, with `w_c=0.5`.\n"
            )
        findings = f"{uniform_note}\n## Complete WP2 comparison\n\n" f"{wp2_table}\n"
    elif record.work_package == "WP4":
        findings = "\n## Complete WP4 sensitivity table\n\n" f"{wp4_table}\n\n" f"{WP4_MISLABEL}\n"
    elif record.work_package == "WP6":
        findings = f"\n{WP6_HEADLINES}\n"
    else:
        findings = ""
    return f"""# {record.name} — results

Status: **complete**
Work package: {record.work_package}
Arm/setting: {record.arm}

## How it was run

```bash
.venv/bin/python tools/run_exact_job.py --run-config {run_dir}/{record.run_yaml}
```

The local launcher used CUDA device 0, seed 42, the {record.dataset_label}, full reference
`{record.reference_path}`, and no training. The run-specific configuration statement is
{record.config_delta}.{note}
## Metrics

| Metric | Value |
|---|---:|
| MRR | {metrics['MRR']:.3f} |
| Hits@1 | {metrics['Hits@1']:.3f} |
| Hits@5 | {metrics['Hits@5']:.3f} |
| Hits@10 | {metrics['Hits@10']:.3f} |

Wall clock: **{wall_clock:.1f} minutes** (`times.txt:Total`).
{findings}

## Full artifacts

Run folder: `{run_dir.resolve()}`

Present: {", ".join(artifacts)}.
"""


def render_consolidated(runs_root: Path, wp2_table: str, wp4_table: str) -> str:
    wp1_headlines = WP1_HEADLINES.replace(
        "`../analysis/` from this run folder",
        "`analysis/`",
    )
    wp6_headlines = WP6_HEADLINES.replace("../analysis/", "analysis/")
    return f"""# Review-response experiment results

Status: **complete** — 14 fresh physical ranking runs across WP1–WP6, with one WP1/WP2 reuse and
one WP4 gamma/tau reuse. WP3 used the specification-approved existing-log harvest because all five
task families were covered; WP5 re-analysed the existing E1 dump and required no new GPU run.

The full methods, interpretation, limitations, and artifact index are in
[`specs/review-response/FINAL_REPORT.md`](../../../specs/review-response/FINAL_REPORT.md), with the
completed post-report work in
[`FINAL_REPORT_ADDENDUM.md`](../../../specs/review-response/FINAL_REPORT_ADDENDUM.md).

{wp1_headlines}

## Complete WP2 comparison

{wp2_table}

The uniform arm uses `|K|=6`: three hierarchy-family keys plus similarity, difference, and
attributes.

## Complete WP4 sensitivity table

{wp4_table}

`e4_gamma_2` is the single physical `(gamma=2, tau=0.5)` run shown in both series.
{WP4_MISLABEL}

WP3's full table is in [`analysis/e3_abstention.csv`](analysis/e3_abstention.csv), with
interpretation in [`analysis/WP3_RESULTS.md`](analysis/WP3_RESULTS.md).

{WP5_HEADLINES}

{wp6_headlines}

Full run and analysis root: `{runs_root.resolve()}`.
"""


def main() -> None:
    args = parse_args()
    wp2_table = render_wp2_table(args.runs_root)
    wp4_table = render_wp4_table(args.runs_root)
    for record in RUNS:
        run_dir = args.runs_root / record.name
        required = [run_dir / "evaluation_results.csv", run_dir / "times.txt"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{record.name}: missing {missing}")
        output = run_dir / "RESULTS.md"
        output.write_text(
            render(record, run_dir, wp2_table, wp4_table),
            encoding="utf-8",
        )
        print(output)
    consolidated = args.runs_root / "RESULTS.md"
    consolidated.write_text(
        render_consolidated(args.runs_root, wp2_table, wp4_table),
        encoding="utf-8",
    )
    print(consolidated)


if __name__ == "__main__":
    main()
