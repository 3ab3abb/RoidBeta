"""Tests for the multi-climber session roster and result recording."""

from roidbeta.session import Session
from roidbeta.scoring.profile import SkillProfile


def _profile(label):
    return SkillProfile(label, 0.7, 0.6, 0.3, 1.0, True, 40.0)


def test_solo_session_has_one_climber():
    s = Session.create(1)
    assert s.is_solo and len(s.climbers) == 1


def test_session_clamps_count_to_available_colors():
    s = Session.create(99)
    assert len(s.climbers) <= 6 and not s.is_solo


def test_advance_walks_the_roster_and_records_per_climber():
    s = Session.create(3)
    assert s.current.label == "Climber 1" and s.position_label == "1 of 3"
    s.record(_profile("Climber 1"))
    assert not s.is_last and s.advance()
    assert s.current.label == "Climber 2"
    s.record(_profile("Climber 2"))
    s.advance()
    s.record(_profile("Climber 3"))
    assert s.is_last and not s.advance()   # no one after the last
    assert [p.label for p in s.results] == ["Climber 1", "Climber 2", "Climber 3"]


def test_record_replaces_on_retry():
    s = Session.create(2)
    s.record(_profile("Climber 1"))
    s.record(SkillProfile("Climber 1", 0.9, 0.9, 0.5, 1.0, True, 20.0))  # retry
    assert len(s.results) == 1 and s.results[0].time_s == 20.0
