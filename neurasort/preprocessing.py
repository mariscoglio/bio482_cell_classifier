from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


def free_whisking_preprocessing_pipe()->Pipeline:
    """Create free whisking pre-processing pipeline"""
    scaler= StandardScaler()
    constant_remover = VarianceThreshold(threshold=0.)

    scaler.set_output(transform="pandas")
    constant_remover.set_output(transform="pandas")

    return Pipeline(
        [
            ("scaling", scaler),
            ("remove_constants", constant_remover),
        ]
    )