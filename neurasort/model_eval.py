import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from copy import deepcopy
from typing import Callable

def fit_models_get_CV_scores(
    models: list,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring: str | dict,
    preprocessing_pipe: Pipeline = None,
    class_weight: dict = None,
    model_naming: Callable = lambda model: model.__class__.__name__,
) -> pd.DataFrame:
    """Fit models and get cross-validation scores."""
    step = cv.get_n_splits(X)
    if not isinstance(scoring, dict):
        scoring = {scoring: scoring}
    test_scorings_column_names = [f"test_{scoring_name}" for scoring_name in scoring.keys()]
    train_scorings_column_names = [f"train_{scoring_name}" for scoring_name in scoring.keys()]
    models_CV_scores = pd.DataFrame(
        columns=[
            "model",
            "fold",
            "fit_time",
            *test_scorings_column_names,
            *train_scorings_column_names
        ],
        index=range(len(models) * step),
    )
    for i, model in enumerate(models):
        if class_weight is not None and hasattr(model, "class_weight"):
            model = model.set_params(class_weight=class_weight)
        model_name = model_naming(model)
        print(f"Evaluating model: {model_name}")
        if preprocessing_pipe is not None:
            pipe = deepcopy(preprocessing_pipe)
            pipe.steps.append(("model_used", model))
        else:
            pipe = model
        model_CV_scores = cross_validate(
            estimator=pipe,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )
        for test_scoring_name in test_scorings_column_names:
            models_CV_scores[test_scoring_name].iloc[
                i * step : (i + 1) * step
            ] = model_CV_scores[test_scoring_name]
        for train_scoring_name in train_scorings_column_names:
            models_CV_scores[train_scoring_name].iloc[
                i * step : (i + 1) * step
            ] = model_CV_scores[train_scoring_name]
        models_CV_scores["fit_time"].iloc[i * step : (i + 1) * step] = model_CV_scores[
            "fit_time"
        ]

        models_CV_scores["model"].iloc[i * step : (i + 1) * step] = model_name
        models_CV_scores["fold"].iloc[i * step : (i + 1) * step] = np.arange(step)

    return models_CV_scores