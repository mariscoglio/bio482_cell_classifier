import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from copy import deepcopy

def fit_models_get_CV_scores(
    models: list,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    scoring : str | dict,
    preprocessing_pipe : Pipeline = None,
) -> pd.DataFrame:
    """Fit models and get cross-validation scores."""
    step = cv.get_n_splits(X)
    if not isinstance(scoring, dict):
        scoring = {scoring: scoring}
    models_CV_scores = pd.DataFrame(
        columns=[
            "model",
            "fold",
            *list(scoring.keys()),
        ],
        index=range(len(models) * step),
    )
    for i, model in enumerate(models):
        model_name = model.__class__.__name__
        print(f"Evaluating model: {model_name}")
        if preprocessing_pipe is not None:
            pipe = deepcopy(preprocessing_pipe)
            pipe.steps.append(('model_used', model))
        else:
            pipe = model
        model_CV_scores = cross_validate(
            estimator=pipe,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
        )
        for scoring_name in scoring.keys():
            models_CV_scores[scoring_name].iloc[i * step : (i + 1) * step] = model_CV_scores[f"test_{scoring_name}"]
        # models_CV_scores["model"].iloc[i * step : (i + 1) * step] = model.regressor.__class__.__name__ # If using a transfro
        
        models_CV_scores["model"].iloc[i * step : (i + 1) * step] = model_name
        models_CV_scores["fold"].iloc[i * step : (i + 1) * step] = np.arange(step)
    return models_CV_scores