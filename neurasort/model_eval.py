import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from copy import deepcopy

def fit_models_get_CV_score(
    models: list,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cv,
    scorings : str | list[str],
    preprocessing_pipe : Pipeline = None,
) -> pd.DataFrame:
    """Fit models and get cross-validation scores."""
    step = cv.get_n_splits(X)
    if not isinstance(scorings, list):
        scorings = [scorings]
    models_CV_scores = pd.DataFrame(
        columns=[
            "model",
            "fold",
            *scorings,
        ],
        index=range(len(models) * step),
    )
    for i, model in enumerate(models):
        if preprocessing_pipe is not None:
            pipe = deepcopy(preprocessing_pipe)
            pipe.steps.append(('model_used', model))
        else:
            pipe = model
        for scoring in scorings:
            model_CV_scores = cross_val_score(
                estimator=pipe,
                X=X,
                y=y,
                cv=cv,
                scoring=scoring,
            )
            models_CV_scores[scoring].iloc[i * step : (i + 1) * step] = model_CV_scores
        models_CV_scores["model"].iloc[i * step : (i + 1) * step] = model.__class__.__name__
        
        models_CV_scores["fold"].iloc[i * step : (i + 1) * step] = np.arange(step)
    return models_CV_scores