"""Lightweight interference classifier — jam / spoof / meacon / clean (pure stdlib).

The 2025 literature leans on CNN+LSTM and XGBoost models; those are accurate but
drag in heavy runtimes. This module keeps the *shape* of that work — an offline
model over engineered features — while staying zero-dependency and fully
deterministic so it ships as pure Python and runs on hardware you own.

Two pieces:

* a tiny **CART decision tree** (:class:`DecisionTree`, Gini splits, depth-limited)
  that can be trained on labelled feature vectors, and
* a bundled **rule-based fallback** (:func:`rule_classify`) so the detector still
  works with no trained model at all.

Features (see :data:`FEATURE_NAMES`) are the engineered signals the other
detectors already expose: C/N0 spread, AGC delta, RAIM residual RMS, integrity-
report density, and teleport rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FEATURE_NAMES = ["cn0_spread", "agc_delta", "residual_rms",
                 "integrity_density", "teleport_rate"]
LABELS = ["clean", "jam", "spoof", "meacon"]


def features_to_vector(feats):
    """Turn a feature dict into an ordered vector (missing keys -> 0.0)."""
    return [float(feats.get(k, 0.0)) for k in FEATURE_NAMES]


# --------------------------------------------------------------------------- #
# CART decision tree
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    label: str | None = None
    dist: dict = field(default_factory=dict)


def _gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _majority(labels):
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return best[0], counts


class DecisionTree:
    """A small, deterministic CART classifier (Gini impurity, depth-limited)."""

    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.n_features = len(FEATURE_NAMES)

    def fit(self, X, y):
        self.n_features = len(X[0]) if X else len(FEATURE_NAMES)
        self.root = self._build(list(X), list(y), 0)
        return self

    def _build(self, X, y, depth):
        label, counts = _majority(y)
        node = _Node(label=label, dist=counts)
        if (depth >= self.max_depth or len(set(y)) <= 1
                or len(y) < self.min_samples):
            return node
        best = self._best_split(X, y)
        if best is None:
            return node
        feat, thr, li, ri = best
        node.feature = feat
        node.threshold = thr
        node.label = None
        node.left = self._build([X[i] for i in li], [y[i] for i in li], depth + 1)
        node.right = self._build([X[i] for i in ri], [y[i] for i in ri], depth + 1)
        return node

    def _best_split(self, X, y):
        n = len(y)
        base = _gini(y)
        best_gain = 1e-12
        best = None
        for f in range(self.n_features):
            vals = sorted(set(row[f] for row in X))
            # candidate thresholds = midpoints between distinct values
            for a, b in zip(vals, vals[1:]):
                thr = (a + b) / 2.0
                li = [i for i in range(n) if X[i][f] <= thr]
                ri = [i for i in range(n) if X[i][f] > thr]
                if not li or not ri:
                    continue
                gini = (len(li) * _gini([y[i] for i in li])
                        + len(ri) * _gini([y[i] for i in ri])) / n
                gain = base - gini
                if gain > best_gain:
                    best_gain = gain
                    best = (f, thr, li, ri)
        return best

    def _walk(self, vec):
        node = self.root
        while node.label is None:
            node = node.left if vec[node.feature] <= node.threshold else node.right
        return node

    def predict_one(self, vec):
        return self._walk(vec).label

    def predict(self, X):
        return [self.predict_one(v) for v in X]

    def predict_proba_one(self, vec):
        node = self._walk(vec)
        tot = sum(node.dist.values()) or 1
        return {lbl: node.dist.get(lbl, 0) / tot for lbl in LABELS}


# --------------------------------------------------------------------------- #
# rule-based fallback
# --------------------------------------------------------------------------- #
def rule_classify(feats):
    """Deterministic rule fallback when no trained tree is available.

    Mirrors the physical signatures: an AGC drop / high residual -> jam; a tight
    C/N0 spread with teleports -> spoof; teleports with moderate residual and
    density -> meacon; otherwise clean.
    """
    cn0_spread = float(feats.get("cn0_spread", 0.0))
    agc_delta = float(feats.get("agc_delta", 0.0))
    residual = float(feats.get("residual_rms", 0.0))
    density = float(feats.get("integrity_density", 0.0))
    teleport = float(feats.get("teleport_rate", 0.0))

    if agc_delta <= -0.2 or (residual > 40 and density > 0.3):
        return "jam"
    if teleport > 0.05 and cn0_spread < 4.0 and residual > 20:
        return "spoof"
    if teleport > 0.02 and 5 <= residual <= 30 and density < 0.2:
        return "meacon"
    return "clean"


class InterferenceClassifier:
    """Trained-tree classifier with graceful rule-based fallback."""

    def __init__(self, tree=None):
        self.tree = tree

    def fit(self, samples, labels, max_depth=5):
        X = [features_to_vector(s) for s in samples]
        self.tree = DecisionTree(max_depth=max_depth).fit(X, labels)
        return self

    def classify(self, feats):
        if self.tree is None or self.tree.root is None:
            return rule_classify(feats)
        return self.tree.predict_one(features_to_vector(feats))

    def proba(self, feats):
        if self.tree is None or self.tree.root is None:
            lbl = rule_classify(feats)
            return {l: (1.0 if l == lbl else 0.0) for l in LABELS}
        return self.tree.predict_proba_one(features_to_vector(feats))
