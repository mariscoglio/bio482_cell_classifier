from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


def free_whisking_preprocessing_pipe(
    columns_ohe: list[str] = ["Cell_tdTomatoExpressing", "Cell_Layer"],
    columns_min_max: list[str] = ["Cell_APThreshold_Slope", "Cell_Depth", "firing_rate", "ap_duration", "std_vm", "fft_low", "fft_high"],
    columns_standadize:list[str] = ["ap_threshold", "mean_vm"],
)->Pipeline:
    """Create free whisking pre-processing pipeline"""
    ohe_scaler = make_column_transformer(
        (OneHotEncoder(sparse_output=False, drop="if_binary"), columns_ohe),
        (MinMaxScaler(), columns_min_max),
        (StandardScaler(), columns_standadize),
        remainder="drop",
        verbose_feature_names_out=False,  # avoid to prepend the preprocessor names
    )
    constant_remover = VarianceThreshold(threshold=0.)
    cell_depth_imputer = IterativeImputer(max_iter=10, random_state=0)

    ohe_scaler.set_output(transform="pandas")
    constant_remover.set_output(transform="pandas")
    cell_depth_imputer.set_output(transform="pandas")

    return Pipeline(
        [
            ("OHE & scaling", ohe_scaler),
            ("remove constants", constant_remover),
            ("impute cell depth", cell_depth_imputer),
        ]
    )