import time
from enum import Enum
from typing import Tuple, Dict

from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.pipeline import Pipeline
from category_encoders.cat_boost import CatBoostEncoder
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from autogluon import TabularPrediction


class TaskNames(Enum):
    classification = "classification_task"
    regression = "regression_task"


class ModelNames(Enum):
    catboost = "catboost"
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    xgboost_with_cat_encoder = "xgboost_with_cat_encoder"
    lightgbm_with_cat_encoder = "lightgbm_with_cat_encoder"


class ModelComparison:
    unknown_category = "Unknown category"
    unknown_numeric_value = -1

    def __init__(self,
                 task_name: TaskNames,
                 cross_validation_n_folds: int,
                 features: pd.DataFrame,
                 target: pd.Series):
        self.task_name = task_name
        self.cross_validation_n_folds = cross_validation_n_folds
        numeric_features = set(features.select_dtypes("number").columns)
        self.categorical_features = list(set(features.columns) - numeric_features)
        self.categorical_features_indices = list(np.where(features.columns.isin(self.categorical_features))[0])

        encoded_categorical_features = {
            categorical_feature:
                LabelEncoder().fit_transform(features[categorical_feature].astype(str).fillna(self.unknown_category))
            for categorical_feature in self.categorical_features
        }
        self.features = features.assign(**encoded_categorical_features) \
            .fillna({numeric_feature: self.unknown_numeric_value
                     for numeric_feature in numeric_features})

        if is_numeric_dtype(target):
            self.target = target
        else:
            self.target = LabelEncoder().fit_transform(target)

    def get_default_models_scores_and_training_time(self) -> Dict:
        return {model_name: self._get_default_model_score_and_training_time(model_name)
                for model_name in self.models_to_compare.keys()}

    def _get_default_model_score_and_training_time(self, model_name: ModelNames) -> Tuple[float, float]:
        model = self.models_to_compare[model_name][self.task_name]

        start_time = time.time()
        cross_val_scores = cross_val_score(model,
                                           self.features,
                                           self.target,
                                           cv=self.cross_validation_n_folds,
                                           n_jobs=-1)
        end_time = time.time()
        return np.mean(cross_val_scores), end_time - start_time

    @property
    def models_to_compare(self) -> Dict[ModelNames, Dict[TaskNames, object]]:
        return {
            ModelNames.catboost: {
                TaskNames.classification: CatBoostClassifier(cat_features=self.categorical_features_indices),
                TaskNames.regression: CatBoostRegressor(cat_features=self.categorical_features_indices)
            },
            ModelNames.lightgbm: {
                TaskNames.classification: LGBMClassifier(categorical_features=self.categorical_features_indices),
                TaskNames.regression: LGBMRegressor(categorical_features=self.categorical_features_indices)
            },
            ModelNames.lightgbm_with_cat_encoder: {
                TaskNames.classification: Pipeline([("encoder", CatBoostEncoder()),
                                                    ("lightgbm", LGBMClassifier())]),
                TaskNames.regression: Pipeline([("encoder", CatBoostEncoder()),
                                                ("lightgbm", LGBMRegressor())]),
            },
            ModelNames.xgboost: {
                TaskNames.classification: XGBClassifier(),
                TaskNames.regression: XGBRegressor()
            },
            ModelNames.xgboost_with_cat_encoder: {
                TaskNames.classification: Pipeline([("encoder", CatBoostEncoder(cols=self.categorical_features)),
                                                    ("xgboost", XGBClassifier())]),
                TaskNames.regression: Pipeline([("encoder", CatBoostEncoder(cols=self.categorical_features)),
                                                ("xgboost", XGBRegressor())]),
            }
        }