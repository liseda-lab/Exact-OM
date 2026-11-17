"""Train-only edge classification with portable, authenticated model artifacts."""

import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .features import FEATURES, RAW, greedy, select
from .io import atomic, digest, export, partial_metrics, prepare, read_frame, references, write_json

THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.98,
    0.99,
    0.995,
]


def training_population(
    data, train, valid=(), *, seed=20261116, folds=5, background_size=8192, membership=None
):
    """Build source/target held-out masks; unknown background is never gold."""
    train_sources = {s for s, _ in train}
    if train_sources & {s for s, _ in valid} or {t for _, t in train} & {t for _, t in valid}:
        raise ValueError("TRAIN and VALID endpoints must be disjoint")
    if not 2 <= folds <= len(train_sources):
        raise ValueError("Fold count must be between two and number of TRAIN sources")
    if background_size < 0:
        raise ValueError("Background sample size cannot be negative")
    order = lambda s: (hashlib.sha256((str(seed) + "\0" + s).encode()).hexdigest(), s)
    foldmap = (
        {s: i % folds for i, s in enumerate(sorted(train_sources, key=order))}
        if membership is None
        else {s: int(membership[s]) for s in train_sources}
    )
    if set(foldmap.values()) != set(range(folds)):
        raise ValueError("Fold membership must represent every configured fold")
    sm, tm = {s: i for i, s in enumerate(data.sources)}, {t: i for i, t in enumerate(data.targets)}
    sf = np.full(len(sm), -1, np.int32)
    tf = np.full(len(tm), -1, np.int32)
    ks, kt = np.zeros(len(sm), bool), np.zeros(len(tm), bool)
    gold = []
    for source, target in train:
        fold = foldmap[source]
        if source in sm:
            ks[sm[source]], sf[sm[source]] = True, fold
        if target in tm:
            kt[tm[target]], tf[tm[target]] = True, fold
        if source in sm and target in tm:
            gold.append(sm[source] * len(tm) + tm[target])
    unknown = sorted(set(data.sources) - train_sources - {s for s, _ in valid}, key=order)
    sampled = unknown[:background_size]
    bg = np.zeros(len(sm), bool)
    for i, source in enumerate(sampled):
        bg[sm[source]], sf[sm[source]] = True, i % folds
    keys = data.src.astype(np.int64) * len(tm) + data.tgt
    return dict(
        keys=keys,
        gold=np.asarray(sorted(gold), np.int64),
        label=np.isin(keys, gold),
        known_s=ks,
        known_t=kt,
        sf=sf,
        tf=tf,
        bg=bg,
        sm=sm,
        tm=tm,
        expansion=len(unknown) / len(sampled) if sampled else 0.0,
        pairs=train,
        foldmap=foldmap,
        background_sources=sampled,
        background_population=len(unknown),
    )


def training_rows(data, population, fold, weak_weight):
    """Select real edges, excluding both held-out endpoint sets."""
    x, src, tgt, d = data.x, data.src, data.tgt, population
    ks, kt = d["known_s"][src], d["known_t"][tgt]
    labelled = ks | (kt & (x[:, FEATURES.index("target_rank")] >= 1 / 5)) | d["label"]
    unknown = ~ks & ~kt & d["bg"][src] & (x[:, FEATURES.index("source_rank")] >= 1 / 3)
    mask = labelled | (unknown if weak_weight else False)
    if fold is not None:
        mask &= (d["sf"][src] != fold) & (d["tf"][tgt] != fold)
    idx = np.flatnonzero(mask)
    labels = d["label"][idx].astype(np.int8)
    weights = np.ones(len(idx))
    weak = unknown[idx] & ~labelled[idx]
    if weak.any():
        counts = np.bincount(src[idx[weak]], minlength=len(d["bg"]))
        weights[weak] = weak_weight * d["expansion"] / counts[src[idx[weak]]]
    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Both fit classes required; too few retrieved TRAIN positives or negatives"
        )
    if fold is not None and (
        (d["sf"][src[idx]] == fold).any() or (d["tf"][tgt[idx]] == fold).any()
    ):
        raise ValueError("Held-out endpoint leaked into fitting")
    return idx, labels, weights


def fit_estimator(x, labels, weights, *, family="hist", seed=20261116, threads=1):
    """Fit the qualified classifier; normalization is fitted on TRAIN only."""
    with threadpool_limits(limits=threads):
        if family == "logistic":
            scaler = StandardScaler().fit(x, sample_weight=weights)
            model = LogisticRegression(
                C=1.0, solver="liblinear", max_iter=300, tol=1e-5, random_state=seed
            ).fit(scaler.transform(x), labels, sample_weight=weights)
        elif family == "hist":
            scaler = None
            model = HistGradientBoostingClassifier(
                max_iter=80,
                max_leaf_nodes=7,
                max_depth=3,
                learning_rate=0.08,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ).fit(x, labels, sample_weight=weights)
        else:
            raise ValueError("Model family must be hist or logistic")
    return scaler, model


def predict(estimator, x, *, threads=1, chunk_size=100000):
    """Bound transformation memory and preserve the qualified float32 scores."""
    if threads < 1 or chunk_size < 1:
        raise ValueError("Prediction threads and chunk size must be positive")
    scaler, model = estimator
    out = np.empty(len(x), np.float32)
    with threadpool_limits(limits=threads):
        for start in range(0, len(x), chunk_size):
            block = np.asarray(x[start : start + chunk_size], dtype=np.float64)
            out[start : start + chunk_size] = model.predict_proba(
                scaler.transform(block) if scaler is not None else block
            )[:, 1]
    if not np.isfinite(out).all():
        raise ValueError("Model produced nonfinite probabilities")
    return out


def save_model(path, estimator):
    """Atomically serialize a locally trusted estimator."""
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer)
    atomic(path, buffer.getvalue())


def fold_curves(prob, data, d, fold, thresholds):
    """Report pooled OOF reference-relative and conservative background objectives."""
    selected = greedy(prob, data.src, data.tgt)
    p, ss, tt = prob[selected], data.src[selected], data.tgt[selected]
    held_s = d["known_s"] & (d["sf"] == fold)
    held_t = d["known_t"] & (d["tf"] == fold)
    ref = {(s, t) for s, t in d["pairs"] if d["foldmap"][s] == fold}
    gold = [
        d["sm"][s] * len(data.targets) + d["tm"][t] for s, t in ref if s in d["sm"] and t in d["tm"]
    ]
    tp = np.isin(d["keys"][selected], gold)
    judged = held_s[ss] | held_t[tt]
    bg = d["bg"][ss] & (d["sf"][ss] == fold)
    return [
        dict(
            fold=fold,
            threshold=float(threshold),
            TP=int(tp[p > threshold].sum()),
            partial_FP=int((judged & ~tp)[p > threshold].sum()),
            known_FP=int((held_s[ss] & ~tp)[p > threshold].sum()),
            weak_background_penalty=float(bg[p > threshold].sum() * d["expansion"]),
            FN=len(ref) - int(tp[p > threshold].sum()),
        )
        for threshold in thresholds
    ]


def choose_threshold(curves, objective):
    """Choose on TRAIN OOF only; the full objective is a conservative surrogate."""
    candidates = []
    for threshold in sorted({row["threshold"] for row in curves}):
        rows = [row for row in curves if row["threshold"] == threshold]
        tp, fn = sum(r["TP"] for r in rows), sum(r["FN"] for r in rows)
        fp = sum(
            (
                r["partial_FP"]
                if objective == "partial"
                else r["known_FP"] + r["weak_background_penalty"]
            )
            for r in rows
        )
        candidates.append(
            dict(
                threshold=threshold,
                TP=tp,
                FP=fp,
                FN=fn,
                P=tp / (tp + fp) if tp + fp else 0.0,
                R=tp / (tp + fn) if tp + fn else 0.0,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            )
        )
    return max(candidates, key=lambda r: (r["F1"], r["P"], r["R"], r["threshold"]))


def fit(
    candidate_path,
    train_path,
    directory,
    *,
    valid_path=None,
    family="hist",
    weak_weight=0.02,
    seed=20261116,
    folds=5,
    background_size=8192,
    threads=1,
    membership_path=None,
    thresholds=THRESHOLDS,
):
    """Fit one declared family and freeze full/partial policies without TEST labels.

    Compatible completed fold models resume. VALID may choose the partial
    assignment method; full uses the TRAIN background surrogate and greedy
    selection. The historical baseline qualification/fallback is not invented
    when no prior model exists.
    """
    if not 0 <= weak_weight <= 1 or threads < 1:
        raise ValueError("Weak weight must be in [0,1], and threads positive")
    if not thresholds or any(not 0 < t < 1 for t in thresholds):
        raise ValueError("Threshold grid must contain values strictly inside (0,1)")
    directory = Path(directory)
    source = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    identity = dict(
        schema=1,
        source=source,
        candidates=digest(candidate_path),
        train=digest(train_path),
        valid=digest(valid_path) if valid_path else None,
        family=family,
        weak_weight=weak_weight,
        seed=seed,
        folds=folds,
        background_size=background_size,
        thresholds=list(thresholds),
        membership=digest(membership_path) if membership_path else None,
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
    )
    manifest = directory / "model.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["training"] != identity:
            raise ValueError("Incompatible completed fit; use a new output directory")
        load(directory)
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "training.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Incompatible checkpoint; use a new output directory")
    write_json(receipt, identity)
    data = prepare(read_frame(candidate_path, columns=["Src", "Tgt", *RAW]))
    train = references(train_path)
    valid = references(valid_path) if valid_path else set()
    membership = json.loads(Path(membership_path).read_text()) if membership_path else None
    d = training_population(
        data,
        train,
        valid,
        seed=seed,
        folds=folds,
        background_size=background_size,
        membership=membership,
    )
    if weak_weight and d["background_population"] and not d["background_sources"]:
        raise ValueError("Weak fitting requires a nonempty background sample")
    curves = []
    final = None
    for fold in [*range(folds), None]:
        stage = directory / ("final" if fold is None else f"fold-{fold}")
        stage.mkdir(exist_ok=True)
        saved = stage / "fit.json"
        if saved.exists():
            info = json.loads(saved.read_text())
            if digest(stage / "model.joblib") != info["sha256"]:
                raise ValueError("Changed checkpoint model")
            estimator = joblib.load(stage / "model.joblib")
        else:
            idx, y, w = training_rows(data, d, fold, weak_weight)
            estimator = fit_estimator(
                np.asarray(data.x[idx], dtype=np.float64),
                y,
                w,
                family=family,
                seed=seed,
                threads=threads,
            )
            save_model(stage / "model.joblib", estimator)
            write_json(
                saved,
                dict(
                    fold=fold,
                    rows=len(idx),
                    positives=int(y.sum()),
                    sha256=digest(stage / "model.joblib"),
                    source_target_holdout_checked=True,
                    row_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),
                ),
            )
        prob = predict(estimator, data.x, threads=threads)
        if fold is None:
            final = prob
            save_model(directory / "model.joblib", estimator)
        else:
            curves.extend(fold_curves(prob, data, d, fold, thresholds))
    policies, validation = {}, []
    for objective in ["partial", "full"]:
        chosen = choose_threshold(curves, objective)
        modes = (
            ["greedy", "winner", "assignment"] if valid and objective == "partial" else ["greedy"]
        )
        alternatives = []
        for mode in modes:
            idx = select(final, data.src, data.tgt, threshold=chosen["threshold"], mode=mode)
            metrics = partial_metrics(data.pairs(idx), valid, train) if valid else None
            alternatives.append(dict(mode=mode, threshold=chosen["threshold"], validation=metrics))
            validation.append(dict(objective=objective, **alternatives[-1]))
        best = (
            max(
                alternatives,
                key=lambda r: (
                    r["validation"]["F1"],
                    r["validation"]["P"],
                    -r["validation"]["selected"],
                ),
            )
            if valid
            else alternatives[0]
        )
        policies[objective] = dict(
            mode=best["mode"],
            threshold=chosen["threshold"],
            selection_reference="VALID" if objective == "partial" and valid else "TRAIN OOF",
            oof=chosen,
        )
    write_json(directory / "oof-curves.json", curves)
    write_json(directory / "validation.json", validation)
    if (
        digest(candidate_path) != identity["candidates"]
        or digest(train_path) != identity["train"]
        or (valid_path and digest(valid_path) != identity["valid"])
    ):
        raise ValueError("Input changed while fitting; no final model published")
    write_json(
        manifest,
        dict(
            schema=1,
            kind="edge_acceptance",
            features=FEATURES,
            model_sha256=digest(directory / "model.joblib"),
            sklearn_version=sklearn.__version__,
            training=identity,
            policies=policies,
            private_test_used=False,
            reference_pairs=len(train),
            reference_retrieved=int(d["label"].sum()),
            background_assumption="Unlabelled sampled winners are a conservative error surrogate, not negative truth",
            background_population=d["background_population"],
            background_sample=len(d["background_sources"]),
            qualification="TRAIN OOF; optional VALID assignment qualification; no historical baseline fallback",
        ),
    )
    return manifest


def load(directory):
    """Load a trusted model bundle after integrity and feature-contract checks.

    Joblib uses pickle. Only open bundles from trusted authors; hashes detect
    corruption, not malicious model authors.
    """
    directory = Path(directory)
    info = json.loads((directory / "model.json").read_text())
    if (
        info.get("schema") != 1
        or info.get("kind") != "edge_acceptance"
        or info.get("features") != FEATURES
    ):
        raise ValueError("Incompatible acceptance feature schema")
    if info.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Model requires scikit-learn " + str(info.get("sklearn_version")))
    if digest(directory / "model.joblib") != info["model_sha256"]:
        raise ValueError("Acceptance model hash changed")
    estimator = joblib.load(directory / "model.joblib")
    if (
        not isinstance(estimator, tuple)
        or len(estimator) != 2
        or getattr(estimator[1], "n_features_in_", None) != len(FEATURES)
    ):
        raise ValueError("Model does not implement the acceptance feature contract")
    return estimator, info


def import_model(path, directory, *, threshold, mode, policy, trusted=False):
    """Wrap a trusted frozen (scaler, estimator) joblib without refitting."""
    if not trusted:
        raise ValueError("Explicit trusted=True required for pickle-based model import")
    select(
        np.array([], dtype=float),
        np.array([], dtype=int),
        np.array([], dtype=int),
        threshold=threshold,
        mode=mode,
    )
    if policy not in {"full", "partial"}:
        raise ValueError("Policy must be full or partial")
    directory = Path(directory)
    if (directory / "model.json").exists():
        raise ValueError("Output already contains a frozen model")
    estimator = joblib.load(path)
    if (
        not isinstance(estimator, tuple)
        or len(estimator) != 2
        or getattr(estimator[1], "n_features_in_", None) != len(FEATURES)
    ):
        raise ValueError("Frozen model does not match the 33-feature acceptance contract")
    save_model(directory / "model.joblib", estimator)
    write_json(
        directory / "model.json",
        dict(
            schema=1,
            kind="edge_acceptance",
            features=FEATURES,
            model_sha256=digest(directory / "model.joblib"),
            sklearn_version=sklearn.__version__,
            imported_model_sha256=digest(path),
            refitted=False,
            policies={
                policy: dict(
                    mode=mode, threshold=threshold, selection_reference="Explicit frozen policy"
                )
            },
        ),
    )
    return directory / "model.json"


def apply(candidate_path, directory, output, *, policy, threads=1):
    """Apply a frozen policy to a complete scored graph without opening labels."""
    estimator, info = load(directory)
    if policy not in info["policies"]:
        raise ValueError("Requested policy absent from model artifact")
    data = prepare(read_frame(candidate_path, columns=["Src", "Tgt", *RAW]))
    prob = predict(estimator, data.x, threads=threads)
    setting = info["policies"][policy]
    idx = select(prob, data.src, data.tgt, threshold=setting["threshold"], mode=setting["mode"])
    return export(
        output,
        data,
        idx,
        prob,
        dict(
            name=policy,
            **setting,
            model_manifest_sha256=digest(Path(directory) / "model.json"),
            candidates_sha256=digest(candidate_path),
        ),
    )
