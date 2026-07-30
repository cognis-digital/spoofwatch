import random

from spoofwatch import classify


def _training_set(seed=0, n=60):
    rng = random.Random(seed)
    samples, labels = [], []
    for _ in range(n):
        samples.append({"cn0_spread": rng.uniform(0, 3), "agc_delta": rng.uniform(-0.05, 0.05),
                        "residual_rms": rng.uniform(0, 8), "integrity_density": rng.uniform(0, 0.08),
                        "teleport_rate": 0.0})
        labels.append("clean")
    for _ in range(n):
        samples.append({"cn0_spread": rng.uniform(0, 3), "agc_delta": rng.uniform(-0.6, -0.3),
                        "residual_rms": rng.uniform(45, 80), "integrity_density": rng.uniform(0.3, 0.6),
                        "teleport_rate": 0.0})
        labels.append("jam")
    for _ in range(n):
        samples.append({"cn0_spread": rng.uniform(0, 2), "agc_delta": rng.uniform(-0.05, 0.05),
                        "residual_rms": rng.uniform(25, 40), "integrity_density": rng.uniform(0, 0.08),
                        "teleport_rate": rng.uniform(0.1, 0.3)})
        labels.append("spoof")
    return samples, labels


def test_features_to_vector_order():
    v = classify.features_to_vector({"teleport_rate": 5, "cn0_spread": 1})
    assert v[0] == 1.0            # cn0_spread first
    assert v[-1] == 5.0           # teleport_rate last
    assert len(v) == len(classify.FEATURE_NAMES)


def test_gini_pure_zero():
    assert classify._gini(["a", "a", "a"]) == 0.0


def test_gini_mixed_positive():
    assert classify._gini(["a", "b"]) > 0.0


def test_decision_tree_fits_and_separates():
    X = [[0, 0], [0, 1], [10, 0], [10, 1]]
    y = ["lo", "lo", "hi", "hi"]
    t = classify.DecisionTree(max_depth=3).fit(X, y)
    assert t.predict_one([0, 0]) == "lo"
    assert t.predict_one([10, 1]) == "hi"


def test_tree_train_accuracy():
    samples, labels = _training_set()
    clf = classify.InterferenceClassifier().fit(samples, labels, max_depth=6)
    correct = sum(1 for s, l in zip(samples, labels) if clf.classify(s) == l)
    assert correct / len(labels) > 0.9


def test_tree_generalizes():
    samples, labels = _training_set(seed=1)
    clf = classify.InterferenceClassifier().fit(samples, labels, max_depth=6)
    # a clear jam-like point
    assert clf.classify({"agc_delta": -0.5, "residual_rms": 60,
                         "integrity_density": 0.5, "cn0_spread": 1, "teleport_rate": 0}) == "jam"


def test_proba_sums_to_one():
    samples, labels = _training_set()
    clf = classify.InterferenceClassifier().fit(samples, labels)
    p = clf.proba({"agc_delta": -0.5, "residual_rms": 60, "integrity_density": 0.5})
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_rule_classify_jam():
    assert classify.rule_classify({"agc_delta": -0.4, "residual_rms": 10}) == "jam"


def test_rule_classify_spoof():
    assert classify.rule_classify({"teleport_rate": 0.1, "cn0_spread": 1.0,
                                   "residual_rms": 30}) == "spoof"


def test_rule_classify_meacon():
    assert classify.rule_classify({"teleport_rate": 0.03, "residual_rms": 15,
                                   "integrity_density": 0.1}) == "meacon"


def test_rule_classify_clean():
    assert classify.rule_classify({"cn0_spread": 1, "agc_delta": 0.0,
                                   "residual_rms": 2, "integrity_density": 0.01,
                                   "teleport_rate": 0.0}) == "clean"


def test_fallback_without_training():
    clf = classify.InterferenceClassifier()   # no tree
    assert clf.classify({"agc_delta": -0.4, "residual_rms": 10}) == "jam"
    p = clf.proba({"agc_delta": -0.4, "residual_rms": 10})
    assert p["jam"] == 1.0


def test_predict_batch():
    X = [[0, 0], [10, 10]]
    y = ["a", "b"]
    t = classify.DecisionTree().fit(X, y)
    assert t.predict([[0, 0], [10, 10]]) == ["a", "b"]
