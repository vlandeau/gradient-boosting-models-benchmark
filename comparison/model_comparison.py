from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd
import time
from catboost import CatBoostClassifier, CatBoostRegressor
from category_encoders import CatBoostEncoder, OrdinalEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from comparison.comparison_datasets import ComparisonDataset, TaskName

CATEGORICAL_FEATURE = "categorical_feature"
FIT_PARAMS = "fit_params"
DEFAULT_PARAMETERS = "default_parameters"
TRAINING_TIME = "training_time"
MODEL_SCORE = "model_score"


class ModelName(str, Enum):
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LIGHTGBM_WITH_CATBOOST_ENCODER = "lightgbm_with_catboost_encoder"
    XGBOOST_WITH_CATBOOST_ENCODER = "xgboost_with_catboost_encoder"
    CATBOOST_ENCODER = "catboost_encoder"
    ORDINAL_ENCODER = "ordinal_encoder"


class ModelComparison:
    unknown_category = "Unknown category"
    unknown_numeric_value = -1

    def __init__(self,
                 comparison_dataset: ComparisonDataset,
                 max_parameters_to_test_in_tuning: int = 25):
        self.max_parameters_to_test_in_tuning = max_parameters_to_test_in_tuning
        self.task_name = comparison_dataset.task
        self.cross_validation_n_folds = comparison_dataset.cross_validation_n_folds
        features = comparison_dataset.features
        numeric_features = set(features.select_dtypes("number").columns)
        self.categorical_features = list(set(features.columns) - numeric_features)
        self.categorical_features_indices = list(np.where(features.columns.isin(self.categorical_features))[0])

        features_with_encoded_dates = self._encode_date_columns_as_int(features)
        self.preprocessed_features = features_with_encoded_dates.assign(**{
            categorical_feature: features_with_encoded_dates[categorical_feature].astype("object").fillna(self.unknown_category)
            for categorical_feature in self.categorical_features
        })

        target = comparison_dataset.target
        if is_numeric_dtype(target):
            self.target = target
        else:
            self.target = LabelEncoder().fit_transform(target)

    def get_models_scores_and_training_time(self) -> Dict[ModelName, Dict[str, object]]:
        return {model_name: self._get_default_model_score_and_training_time(model_name)
                for model_name in self.models_to_compare.keys()}

    def _get_default_model_score_and_training_time(self, model_name: ModelName) -> Dict[str, object]:
        model = self.models_to_compare[model_name][self.task_name]

        start_time = time.time()
        cross_val_scores = cross_val_score(model,
                                           self.preprocessed_features,
                                           self.target,
                                           cv=KFold(self.cross_validation_n_folds,
                                                    shuffle=True),
                                           n_jobs=-1,
                                           fit_params=self.models_to_compare[model_name].get(FIT_PARAMS, None))
        end_time = time.time()
        return {MODEL_SCORE: np.mean(cross_val_scores),
                TRAINING_TIME: end_time - start_time}

    @property
    def models_to_compare(self) -> Dict[ModelName, Dict]:
        lightgbm_step_categorical_features_params = f"{ModelName.LIGHTGBM.value}__{CATEGORICAL_FEATURE}"
        return {
            ModelName.CATBOOST: {
                TaskName.CLASSIFICATION: Pipeline([
                    (ModelName.CATBOOST.value,
                     CatBoostClassifier(cat_features=self.categorical_features_indices, verbose=0))]),
                TaskName.REGRESSION: Pipeline([
                    (ModelName.CATBOOST.value,
                     CatBoostRegressor(cat_features=self.categorical_features_indices, verbose=0))])
            },
            ModelName.LIGHTGBM: {
                TaskName.CLASSIFICATION: Pipeline([
                    (ModelName.ORDINAL_ENCODER, OrdinalEncoder()),
                    (ModelName.LIGHTGBM.value, LGBMClassifier())]),
                TaskName.REGRESSION: Pipeline([
                    (ModelName.ORDINAL_ENCODER, OrdinalEncoder()),
                    (ModelName.LIGHTGBM.value, LGBMRegressor())]),
                FIT_PARAMS: {lightgbm_step_categorical_features_params: self.categorical_features}
            },
            ModelName.LIGHTGBM_WITH_CATBOOST_ENCODER: {
                TaskName.CLASSIFICATION: Pipeline(
                    [(ModelName.CATBOOST_ENCODER.value, CatBoostEncoder(cols=self.categorical_features,
                                                                        verbose=0)),
                     (ModelName.LIGHTGBM.value, LGBMClassifier())]),
                TaskName.REGRESSION: Pipeline(
                    [(ModelName.CATBOOST_ENCODER.value, CatBoostEncoder(cols=self.categorical_features,
                                                                        verbose=-1)),
                     (ModelName.LIGHTGBM.value, LGBMRegressor())])
            },
            ModelName.XGBOOST_WITH_CATBOOST_ENCODER: {
                TaskName.CLASSIFICATION: Pipeline(
                    [(ModelName.CATBOOST_ENCODER.value, CatBoostEncoder(cols=self.categorical_features, verbose=0)),
                     (ModelName.XGBOOST.value, XGBClassifier())]),
                TaskName.REGRESSION: Pipeline([
                    (ModelName.CATBOOST_ENCODER.value, CatBoostEncoder(cols=self.categorical_features, verbose=-1)),
                    (ModelName.XGBOOST.value, XGBRegressor())])
            },
            ModelName.XGBOOST: {
                TaskName.CLASSIFICATION: Pipeline([
                    (ModelName.ORDINAL_ENCODER, OrdinalEncoder()),
                    (ModelName.XGBOOST.value, XGBClassifier())]),
                TaskName.REGRESSION: Pipeline([
                    (ModelName.ORDINAL_ENCODER, OrdinalEncoder()),
                    (ModelName.XGBOOST.value, XGBRegressor())])
            }
        }

    @staticmethod
    def _encode_date_columns_as_int(df: pd.DataFrame,
                                    number_of_rows_to_use=10,
                                    date_format_regex=r'\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}') \
            -> pd.DataFrame:
        date_columns_mask = df.sample(number_of_rows_to_use).astype(str).apply(
            lambda x: x.str.match(date_format_regex).all())
        date_columns = date_columns_mask[date_columns_mask].index.tolist()
        columns_mapping = {column_name: pd.to_datetime(df[column_name]).apply(lambda x: time.mktime(x.timetuple()))
                           for column_name in date_columns}
        return df.assign(**columns_mapping)
