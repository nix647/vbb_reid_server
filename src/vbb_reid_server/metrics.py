"""Scoring machine labels against ground truth (human labels or adjudication).

Labels are clustered (bulk naming of board classes, plan §3b D1), so precision is
also bootstrapped over days, never only per sighting."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .decide import MachineLabel


@dataclass
class Tally:
    n: int = 0
    correct: int = 0
    wrong: int = 0
    flagged: int = 0
    abstain: int = 0
    wrong_ids: list[str] = field(default_factory=list)

    def add(self, outcome: str, sid: str | None = None) -> None:
        self.n += 1
        setattr(self, outcome, getattr(self, outcome) + 1)
        if outcome == "wrong" and sid:
            self.wrong_ids.append(sid)

    @property
    def labeled(self) -> int:
        return self.correct + self.wrong

    @property
    def precision(self) -> float | None:
        return self.correct / self.labeled if self.labeled else None

    @property
    def coverage(self) -> float | None:
        return self.labeled / self.n if self.n else None

    def merge(self, other: "Tally") -> None:
        for f in ("n", "correct", "wrong", "flagged", "abstain"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.wrong_ids.extend(other.wrong_ids)


def outcome(label: MachineLabel | None, truth_key: str | None) -> str:
    """truth_key: identity key, or ``"stranger"``. A ``flag`` is neither right nor wrong."""
    if label is None:
        return "abstain"
    if label.state != "known":
        return "flagged"
    return "correct" if label.person_key == truth_key else "wrong"


def bootstrap_precision(per_day: dict[str, Tally], iters: int = 2000, seed: int = 7,
                        q: float = 0.025) -> float | None:
    """Lower quantile of precision when days are resampled with replacement."""
    days = [d for d, t in per_day.items() if t.labeled > 0]
    if not days:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        c = w = 0
        for _ in days:
            t = per_day[rng.choice(days)]
            c += t.correct
            w += t.wrong
        if c + w:
            vals.append(c / (c + w))
    if not vals:
        return None
    vals.sort()
    return vals[max(0, int(q * len(vals)) - 1)]
