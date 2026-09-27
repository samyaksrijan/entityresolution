"""Native CatBoost training with explicit ordered inputs and inner-fold stopping."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, Pool

from entity_resolution.feature_schema import CATEGORICAL_FEATURES
from entity_resolution.training_schema import model_inputs


def fit(frame, labels, parameters, seed, evaluation=None):
    if len(set(labels)) != 2:
        raise ValueError("CatBoost training requires retrieved positives and negatives")
    options = dict(parameters)
    options.update(
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        loss_function="Logloss",
        task_type="CPU",
    )
    patience = options.pop("early_stopping_rounds", 30)
    model = CatBoostClassifier(**options)
    pool = Pool(model_inputs(frame), label=labels, cat_features=list(CATEGORICAL_FEATURES))
    kwargs = {}
    if evaluation is not None:
        val, val_labels = evaluation
        if len(val) and len(set(val_labels)) == 2:
            kwargs.update(
                eval_set=Pool(
                    model_inputs(val), label=val_labels, cat_features=list(CATEGORICAL_FEATURES)
                ),
                early_stopping_rounds=patience,
                use_best_model=True,
            )
    model.fit(pool, **kwargs)
    return model


def predict(model, frame):
    scores = model.predict_proba(model_inputs(frame))[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("invalid model probabilities")
    return scores
