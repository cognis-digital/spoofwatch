from spoofwatch import confidence
from spoofwatch import doppler, skygeom, constellation, clockbias, aoa, signal
from spoofwatch.ecef import geodetic_to_ecef, sat_ecef_from_azel


# --------------------------------------------------------------------------- #
# score()
# --------------------------------------------------------------------------- #
def test_no_evidence_full_trust():
    c = confidence.score({})
    assert c.trust == 1.0
    assert c.spoof_probability == 0.0
    assert c.level == "trusted"


def test_single_strong_channel_lowers_trust():
    c = confidence.score({"raim": 1.0})
    assert c.spoof_probability == 1.0
    assert c.trust == 0.0
    assert c.level == "denied"


def test_weight_scales_contribution():
    strong = confidence.score({"raim": 0.5})       # weight 1.0
    weak = confidence.score({"skygeom": 0.5})       # weight 0.7
    assert strong.spoof_probability > weak.spoof_probability


def test_multiple_channels_combine_upward():
    one = confidence.score({"raim": 0.5})
    two = confidence.score({"raim": 0.5, "doppler": 0.5})
    assert two.spoof_probability > one.spoof_probability


def test_noisy_or_never_exceeds_one():
    c = confidence.score({"raim": 1.0, "doppler": 1.0, "aoa": 1.0,
                          "constellation": 1.0})
    assert c.spoof_probability <= 1.0
    assert c.trust >= 0.0


def test_trust_bands():
    assert confidence.score({}).level == "trusted"
    assert confidence.score({"raim": 0.3}).level in ("trusted", "degraded")
    assert confidence.score({"raim": 0.5}).level in ("degraded", "suspect")
    assert confidence.score({"raim": 1.0}).level == "denied"


def test_protection_level_inflates():
    clean = confidence.score({})
    spoofed = confidence.score({"raim": 1.0})
    assert spoofed.protection_level_m > clean.protection_level_m


def test_nominal_protection_level():
    c = confidence.score({})
    assert c.protection_level_m == confidence.NOMINAL_PL_M


def test_contributions_recorded():
    c = confidence.score({"raim": 0.8, "doppler": 0.4})
    assert set(c.contributions.keys()) == {"raim", "doppler"}


def test_top_channel_is_strongest():
    c = confidence.score({"raim": 0.9, "skygeom": 0.3})
    assert c.top_channel == "raim"


def test_channels_sorted():
    c = confidence.score({"doppler": 0.2, "aoa": 0.3})
    assert c.channels == ["aoa", "doppler"]


def test_probabilities_clipped():
    c = confidence.score({"raim": 5.0})            # out-of-range prob clipped to 1
    assert c.spoof_probability == 1.0
    c2 = confidence.score({"raim": -3.0})
    assert c2.spoof_probability == 0.0


def test_unknown_channel_default_weight():
    c = confidence.score({"mystery": 1.0})
    assert c.spoof_probability == 1.0            # default weight 1.0


def test_custom_weights_override():
    c = confidence.score({"raim": 1.0}, weights={"raim": 0.5})
    assert c.spoof_probability == 0.5


def test_custom_nominal_pl():
    c = confidence.score({}, nominal_pl_m=50.0)
    assert c.protection_level_m == 50.0


def test_is_dataclass():
    c = confidence.score({"raim": 0.5})
    assert isinstance(c, confidence.PNTConfidence)


# --------------------------------------------------------------------------- #
# from_detectors()
# --------------------------------------------------------------------------- #
def test_from_detectors_all_none_full_trust():
    c = confidence.from_detectors()
    assert c.trust == 1.0
    assert c.channels == []


def test_from_detectors_raim_fault():
    c = confidence.from_detectors(raim={"fault": True, "confidence": 0.8})
    assert "raim" in c.contributions
    assert c.spoof_probability > 0.0


def test_from_detectors_raim_no_fault_ignored():
    c = confidence.from_detectors(raim={"fault": False, "confidence": 0.8})
    assert c.contributions.get("raim", 0.0) == 0.0


def test_from_detectors_signal_summary():
    summ = {"peak_spoof_score": 0.9, "peak_jam_score": 0.1}
    c = confidence.from_detectors(signal_summary=summ)
    assert c.contributions["signal"] > 0.0


def test_from_detectors_signal_takes_max():
    summ = {"peak_spoof_score": 0.2, "peak_jam_score": 0.7}
    c = confidence.from_detectors(signal_summary=summ)
    # 0.7 * weight 0.9
    assert abs(c.contributions["signal"] - 0.63) < 1e-6


def test_from_detectors_constellation_divergence():
    cc = {"divergence": True, "confidence": 0.6}
    c = confidence.from_detectors(constellation=cc)
    assert c.contributions["constellation"] > 0.0


def test_from_detectors_constellation_no_divergence():
    cc = {"divergence": False, "confidence": 0.0}
    c = confidence.from_detectors(constellation=cc)
    assert c.contributions.get("constellation", 0.0) == 0.0


def test_from_detectors_clockbias_summary():
    summ = {"time_steps": 1, "events": [{"type": "time_step", "confidence": 0.7}]}
    c = confidence.from_detectors(clockbias=summ)
    assert c.contributions["clockbias"] > 0.0


def test_from_detectors_clockbias_list():
    events = [{"type": "time_step", "confidence": 0.5}]
    c = confidence.from_detectors(clockbias=events)
    assert c.contributions["clockbias"] > 0.0


def test_from_detectors_clockbias_empty():
    c = confidence.from_detectors(clockbias={"time_steps": 0, "events": []})
    assert c.contributions["clockbias"] == 0.0


def test_from_detectors_aoa_single_source():
    c = confidence.from_detectors(aoa={"single_source": True, "confidence": 0.9})
    assert c.contributions["aoa"] > 0.0


def test_from_detectors_aoa_clean():
    c = confidence.from_detectors(aoa={"single_source": False, "confidence": 0.0})
    assert c.contributions.get("aoa", 0.0) == 0.0


def test_from_detectors_doppler_dataclass():
    r = doppler.DopplerResult(available=True, n_sats=5, inconsistent=True,
                              confidence=0.7)
    c = confidence.from_detectors(doppler=r)
    assert c.contributions["doppler"] > 0.0


def test_from_detectors_doppler_static():
    r = doppler.DopplerResult(available=True, n_sats=5, static_spoofer=True,
                              confidence=0.6)
    c = confidence.from_detectors(doppler=r)
    assert c.contributions["doppler"] > 0.0


def test_from_detectors_doppler_clean():
    r = doppler.DopplerResult(available=True, n_sats=5, inconsistent=False,
                              confidence=0.0)
    c = confidence.from_detectors(doppler=r)
    assert c.contributions.get("doppler", 0.0) == 0.0


def test_from_detectors_skygeom_suspect():
    c = confidence.from_detectors(skygeom={"suspect": True, "confidence": 0.5})
    assert c.contributions["skygeom"] > 0.0


def test_from_detectors_kalman_reject_fraction():
    events = [{"accepted": True}, {"accepted": False}, {"accepted": False},
              {"accepted": True}]
    c = confidence.from_detectors(kalman_events=events)
    assert abs(c.contributions["kalman"] - 0.5 * confidence.DEFAULT_WEIGHTS["kalman"]) < 1e-6


def test_from_detectors_kalman_all_accepted():
    events = [{"accepted": True}, {"accepted": True}]
    c = confidence.from_detectors(kalman_events=events)
    assert c.contributions["kalman"] == 0.0


def test_from_detectors_teleport_and_colocation():
    c = confidence.from_detectors(teleport_rate=0.3, colocation=0.4)
    assert "teleport" in c.contributions and "colocation" in c.contributions


def test_from_detectors_multi_channel_corroboration():
    c = confidence.from_detectors(
        raim={"fault": True, "confidence": 0.7},
        aoa={"single_source": True, "confidence": 0.8},
        doppler=doppler.DopplerResult(available=True, n_sats=6,
                                      static_spoofer=True, confidence=0.6),
    )
    # three independent channels -> high spoof probability, denied/suspect
    assert c.spoof_probability > 0.8
    assert c.level in ("suspect", "denied")


# --------------------------------------------------------------------------- #
# end-to-end: real detectors -> confidence
# --------------------------------------------------------------------------- #
def test_end_to_end_clean_scene_trusted():
    # genuine sky geometry + genuine doppler -> trusted
    rx_lat, rx_lon = 59.0, 18.0
    rx = geodetic_to_ecef(rx_lat, rx_lon)
    sky = [skygeom.SatGeom(f"G{i}", az, el) for i, (az, el) in
           enumerate([(45, 20), (135, 35), (225, 25), (315, 40), (0, 70)])]
    sky_res = skygeom.check(sky)

    dobs = []
    for i, (az, el) in enumerate([(45, 20), (135, 35), (225, 25), (315, 40)]):
        pos = sat_ecef_from_azel(rx_lat, rx_lon, az, el)
        vel = (300.0, -200.0, 100.0)
        pred = doppler.expected_doppler_hz(rx, (0, 0, 0), pos, vel)
        dobs.append(doppler.DopplerObs(f"G{i}", pos, vel, pred))
    dop_res = doppler.check(dobs, rx)

    c = confidence.from_detectors(skygeom=sky_res, doppler=dop_res)
    assert c.level == "trusted"
    assert c.trust > 0.9


def test_end_to_end_spoofed_scene_denied():
    rx_lat, rx_lon = 59.0, 18.0
    rx = geodetic_to_ecef(rx_lat, rx_lon)
    # clustered sky (single-source) + static doppler
    sky = [skygeom.SatGeom(f"S{i}", 90 + i * 0.4, 45 + i * 0.3) for i in range(6)]
    sky_res = skygeom.check(sky)

    dobs = []
    for i, (az, el) in enumerate([(45, 20), (135, 35), (225, 25), (315, 40)]):
        pos = sat_ecef_from_azel(rx_lat, rx_lon, az, el)
        vel = (600.0, -400.0, 200.0)
        dobs.append(doppler.DopplerObs(f"G{i}", pos, vel, 0.0))  # collapsed doppler
    dop_res = doppler.check(dobs, rx)

    c = confidence.from_detectors(skygeom=sky_res, doppler=dop_res,
                                  raim={"fault": True, "confidence": 0.9})
    assert c.trust < 0.4
    assert c.level in ("suspect", "denied")
