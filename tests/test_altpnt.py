from spoofwatch import altpnt


def test_leo_available_midlat():
    assert altpnt.leo_soo_available(55.0) is True


def test_leo_unavailable_high_lat():
    assert altpnt.leo_soo_available(85.0) is False


def test_eloran_in_coverage():
    cov = altpnt.eloran_coverage(54.9, -3.0)   # near Anthorn (UK)
    assert cov is not None
    assert cov["station"] == "Anthorn (UK)"
    assert cov["signal_margin"] > 0.9


def test_eloran_out_of_coverage():
    cov = altpnt.eloran_coverage(-33.9, 18.4)  # Cape Town — no station near
    assert cov is None


def test_eloran_nearest_wins():
    cov = altpnt.eloran_coverage(54.808, 8.293)   # exactly at Sylt
    assert cov["station"] == "Sylt (DE)"
    assert cov["distance_km"] < 1.0


def test_availability_good_in_europe():
    a = altpnt.availability(54.9, -3.0)
    assert a["leo_soo"]["available"] is True
    assert a["eloran"] is not None
    assert a["resilience"] in ("moderate", "good")
    assert a["fallback_modes"] >= 2


def test_availability_reports_coast():
    a = altpnt.availability(54.9, -3.0)
    assert a["inertial_coast_s"] > 0


def test_availability_remote_lower_resilience():
    remote = altpnt.availability(-40.0, -120.0)   # mid South-Pacific
    assert remote["eloran"] is None
    # LEO still available at that latitude, so at least LEO + coast
    assert remote["fallback_modes"] <= 2


def test_availability_high_lat_no_leo():
    a = altpnt.availability(85.0, 0.0)
    assert a["leo_soo"]["available"] is False


def test_custom_stations():
    stations = [{"name": "TEST", "lat": 0.0, "lon": 0.0, "coverage_km": 500}]
    cov = altpnt.eloran_coverage(1.0, 1.0, stations=stations)
    assert cov["station"] == "TEST"


def test_leo_accuracy_field():
    a = altpnt.availability(55.0, 20.0)
    assert a["leo_soo"]["nominal_accuracy_m_2d"] == 3.6
