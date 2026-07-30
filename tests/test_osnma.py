import pytest

from spoofwatch import osnma


def test_status_normalization():
    r = osnma.OsnmaReport("X", 1.0, "auth-failed")
    assert r.status == osnma.AUTH_FAILED
    r2 = osnma.OsnmaReport("Y", 1.0, "verified")
    assert r2.status == osnma.VERIFIED


def test_invalid_status_raises():
    with pytest.raises(ValueError):
        osnma.OsnmaReport("X", 1.0, "maybe")


def test_auth_failed_ids():
    reps = [osnma.OsnmaReport("A", 1, "auth-failed"),
            osnma.OsnmaReport("B", 1, "verified"),
            osnma.OsnmaReport("C", 2, "not-verified")]
    assert osnma.auth_failed_ids(reps) == {"A"}


def test_correlate_elevates_overlap():
    spoofs = [{"type": "teleport", "ids": {"A"}, "confidence": 0.7,
               "point": (1, 2), "n": 1}]
    reps = [osnma.OsnmaReport("A", 1, "auth-failed")]
    out = osnma.correlate(spoofs, reps)
    assert out[0]["confidence"] >= osnma.CORROBORATED_CONFIDENCE
    assert out[0]["osnma"]["corroborated"] is True
    assert out[0]["osnma"]["auth_failed_ids"] == ["A"]


def test_correlate_no_overlap_unchanged():
    spoofs = [{"type": "teleport", "ids": {"A"}, "confidence": 0.7,
               "point": (1, 2), "n": 1}]
    reps = [osnma.OsnmaReport("Z", 1, "auth-failed")]
    out = osnma.correlate(spoofs, reps)
    assert out[0]["confidence"] == 0.7
    assert "osnma" not in out[0]


def test_correlate_does_not_mutate_input():
    spoofs = [{"type": "teleport", "ids": {"A"}, "confidence": 0.7,
               "point": (1, 2), "n": 1}]
    reps = [osnma.OsnmaReport("A", 1, "auth-failed")]
    osnma.correlate(spoofs, reps)
    assert spoofs[0]["confidence"] == 0.7
    assert "osnma" not in spoofs[0]


def test_standalone_auth_events():
    reps = [osnma.OsnmaReport("A", 1, "auth-failed"),
            osnma.OsnmaReport("A", 5, "auth-failed"),
            osnma.OsnmaReport("B", 3, "verified")]
    ev = osnma.standalone_auth_events(reps)
    assert len(ev) == 1
    assert ev[0]["ids"] == {"A"}
    assert ev[0]["n_failures"] == 2
    assert ev[0]["first_ts"] == 1 and ev[0]["last_ts"] == 5


def test_standalone_confidence_grows():
    one = osnma.standalone_auth_events([osnma.OsnmaReport("A", 1, "auth-failed")])
    many = osnma.standalone_auth_events(
        [osnma.OsnmaReport("A", t, "auth-failed") for t in range(5)])
    assert many[0]["confidence"] >= one[0]["confidence"]


def test_no_events_when_all_verified():
    reps = [osnma.OsnmaReport("A", 1, "verified"),
            osnma.OsnmaReport("B", 2, "not-verified")]
    assert osnma.standalone_auth_events(reps) == []
    assert osnma.auth_failed_ids(reps) == set()
