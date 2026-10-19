"""Predeclared execution gates and the go/no-go rule (plan pp.5, 11-12).

Gates are *project thresholds*, not expected scientific findings. Each returns a
``GateResult`` that is written to the run directory; a failed gate stops the
stage that depends on it.

Go/no-go (plan p.11): proceed to costly scaling only if the paired confidence
interval on the predeclared primary comparison favours the method, the gain is
practically useful (optional planning target: >= 5 TPR points at the same
achieved FPR), and worst-group false alarms and utility remain acceptable.

``DecisionLog`` keeps the dated decision log the plan asks for. Changing the
primary metric after seeing an unfavourable result must be logged as
exploratory - the log refuses to silently overwrite the primary.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float | int | str | None
    threshold: float | int | str | None
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------ infrastructure
def gate_hook_agreement(max_abs_diff: float, tol: float = 1e-3) -> GateResult:
    """Hooked residual equals a manual HF forward (output_hidden_states[l+1])."""
    return GateResult("infrastructure/hook_agreement", bool(max_abs_diff <= tol), float(max_abs_diff), tol,
                      "hook capture vs manual forward, same dtype and inputs")


def gate_determinism(max_abs_diff: float, tol: float = 1e-5) -> GateResult:
    return GateResult("infrastructure/determinism", bool(max_abs_diff <= tol), float(max_abs_diff), tol,
                      "two identical extraction runs")


# --------------------------------------------------------------- label validity
def gate_t_validity(rate: float) -> GateResult:
    """T edits must preserve latent state and decision on every root they apply to."""
    return GateResult("labels/t_validity", bool(rate >= 1.0 - 1e-12), float(rate), 1.0,
                      "programmatic latent-state + oracle-decision equivalence")


def gate_u_validity(rate: float, threshold: float = 0.8) -> GateResult:
    return GateResult("labels/u_validity", bool(rate >= threshold), float(rate), threshold,
                      "invalid U edits are masked; report rate per branch")


def gate_parse_roundtrip(failures: int) -> GateResult:
    return GateResult("labels/candidate_parse_roundtrip", failures == 0, int(failures), 0,
                      "every candidate string parses back to its slot")


def gate_metadata_shortcut(auroc: float, max_auroc: float = 0.6) -> GateResult:
    """A metadata-only classifier must not predict violations (leakage audit p.9)."""
    return GateResult("labels/metadata_shortcut", bool(auroc <= max_auroc), float(auroc), max_auroc,
                      "suspicious performance requires investigation before any detector result")


# ------------------------------------------------------------------- competence
def gate_schema_validity(rate: float, threshold: float = 0.90) -> GateResult:
    return GateResult("competence/schema_validity", bool(rate >= threshold), float(rate), threshold,
                      "engineering gate (plan p.5), not a finding")


def gate_routine_success(rate: float, threshold: float = 0.80) -> GateResult:
    return GateResult("competence/routine_success", bool(rate >= threshold), float(rate), threshold,
                      "engineering gate (plan p.5), not a finding")


def gate_min_count(name: str, count: int, minimum: int) -> GateResult:
    """Enough screened violations / normals per stratum to fill the quotas."""
    return GateResult(f"pool/{name}", count >= minimum, int(count), int(minimum))


# ---------------------------------------------------------------- split hygiene
def gate_split_hygiene(errors: list[str]) -> GateResult:
    return GateResult("splits/hygiene", not errors, len(errors), 0, "; ".join(errors[:5]))


# ------------------------------------------------------------------- go / no-go
@dataclass(frozen=True)
class GoNoGo:
    decision: str  # "go" | "no-go"
    reasons: tuple[str, ...]
    gates: tuple[GateResult, ...]

    def as_dict(self) -> dict:
        return {"decision": self.decision, "reasons": list(self.reasons),
                "gates": [g.as_dict() for g in self.gates]}


def go_no_go(
    primary_ci_low: float,
    primary_gain: float,
    worst_group_fpr: float,
    nominal_fpr: float,
    practical_gain: float = 0.05,
    max_worst_group_ratio: float = 3.0,
    utility_cost: float | None = None,
    max_utility_cost: float = 0.05,
) -> GoNoGo:
    """Apply the plan's go/no-go rule to the primary paired comparison.

    primary_ci_low: lower end of the paired CI on (method - baseline) TPR at the
                    calibrated primary threshold; must be > 0.
    primary_gain:   point estimate; must be >= ``practical_gain``.
    worst_group_fpr <= max_worst_group_ratio * nominal_fpr.
    utility_cost (if measured) <= max_utility_cost.
    """
    gates = [
        GateResult("go/ci_favours_method", bool(primary_ci_low > 0), primary_ci_low, 0.0),
        GateResult("go/practically_useful", bool(primary_gain >= practical_gain), primary_gain, practical_gain,
                   "decision criterion, not a predicted effect"),
        GateResult("go/worst_group_fpr", bool(worst_group_fpr <= max_worst_group_ratio * nominal_fpr),
                   worst_group_fpr, max_worst_group_ratio * nominal_fpr),
    ]
    if utility_cost is not None:
        gates.append(GateResult("go/utility_cost", bool(utility_cost <= max_utility_cost), utility_cost, max_utility_cost))
    failed = tuple(f"{g.name}: value={g.value} threshold={g.threshold}" for g in gates if not g.passed)
    return GoNoGo("go" if not failed else "no-go", failed or ("all go criteria met",), tuple(gates))


# ------------------------------------------------------------------ decision log
class DecisionLog:
    """Append-only, dated JSONL decision log.

    The first entry of kind ``primary`` fixes the primary comparison. A later
    ``primary`` entry with different content raises unless
    ``exploratory=True``, in which case it is recorded as exploratory and the
    original primary remains the reported one.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, kind: str, content: dict, exploratory: bool = False, note: str = "") -> dict:
        prior = [e for e in self.entries() if e["kind"] == "primary" and not e.get("exploratory")]
        if kind == "primary" and prior and prior[0]["content"] != content and not exploratory:
            raise ValueError(
                "primary comparison already declared; a change after observing results "
                "must be logged with exploratory=True"
            )
        entry = {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "exploratory": bool(exploratory),
            "content": content,
            "note": note,
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry


def write_gates(gates: list[GateResult], path: str | Path) -> bool:
    """Write gate results; return True iff all passed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([g.as_dict() for g in gates], indent=1))
    return all(g.passed for g in gates)
