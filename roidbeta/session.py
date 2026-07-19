"""Climbing session: solo, or several climbers compared on one shared route.

A session is a roster of climbers (each with a label and a color) who take turns
on the same frozen route. Each finished attempt contributes a SkillProfile, and
at the end the profiles are compared on the skill graph and leaderboards.

Solo is just a session of one, so the classic single-climber experience is the
same flow with the comparison step skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .display.skillgraph import CLIMBER_COLORS
from .scoring import SkillProfile


@dataclass(frozen=True)
class Climber:
    label: str
    color: tuple[int, int, int]


@dataclass
class Session:
    """A roster of climbers on one route, plus each climber's result."""

    climbers: list[Climber]
    index: int = 0
    results: list[SkillProfile] = field(default_factory=list)

    @classmethod
    def create(cls, count: int) -> "Session":
        count = max(1, min(count, len(CLIMBER_COLORS)))
        climbers = [
            Climber(f"Climber {i + 1}", CLIMBER_COLORS[i % len(CLIMBER_COLORS)])
            for i in range(count)
        ]
        return cls(climbers=climbers)

    @property
    def is_solo(self) -> bool:
        return len(self.climbers) == 1

    @property
    def current(self) -> Climber:
        return self.climbers[self.index]

    @property
    def is_last(self) -> bool:
        return self.index >= len(self.climbers) - 1

    @property
    def position_label(self) -> str:
        return f"{self.index + 1} of {len(self.climbers)}"

    def record(self, profile: SkillProfile) -> None:
        """Store the current climber's result (replacing it if they retried)."""
        while len(self.results) <= self.index:
            self.results.append(profile)
        self.results[self.index] = profile

    def advance(self) -> bool:
        """Move to the next climber. Returns True if there is one, else False."""
        if self.is_last:
            return False
        self.index += 1
        return True
