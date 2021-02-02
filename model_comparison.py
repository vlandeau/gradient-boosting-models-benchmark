import time
from enum import Enum
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from category_encoders.cat_boost import CatBoostEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


class TaskName(Enum):
    classification = "classification"
    regression = "regression"


class ModelName(Enum):
    catboost = "catboost"
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    xgboost_with_cat_encoder = "xgboost_with_cat_encoder"
    lightgbm_with_cat_encoder = "lightgbm_with_cat_encoder"
    encoder = "encoder"


class ModelComparison:
    unknown_category = "Unknown category"
    unknown_numeric_value = -1

    def __init__(self,
                 task_name: TaskName,
                 cross_validation_n_folds: int,
                 features: pd.DataFrame,
                 target: pd.Series,
                 max_parameters_to_test_in_tuning: int = 50):
        self.max_parameters_to_test_in_tuning = max_parameters_to_test_in_tuning
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

    def _get_default_model_score_and_training_time(self, model_name: ModelName) -> Tuple[float, float]:
        model = self.models_to_compare[model_name][self.task_name]

        start_time = time.time()
        cross_val_scores = cross_val_score(model,
                                           self.features,
                                           self.target,
                                           cv=KFold(self.cross_validation_n_folds,
                                                    shuffle=True),
                                           n_jobs=-1)
        end_time = time.time()
        return np.mean(cross_val_scores), end_time - start_time

    @property
    def models_to_compare(self) -> Dict[ModelName, Dict[TaskName, Pipeline]]:
        return {
            ModelName.catboost: {
                TaskName.classification: Pipeline([(ModelName.catboost.value,
                                                    CatBoostClassifier(cat_features=self.categorical_features_indices,
                                                                       verbose=0))]),
                TaskName.regression: Pipeline([(ModelName.catboost.value,
                                                CatBoostRegressor(cat_features=self.categorical_features_indices,
                                                                  verbose=0))])
            },
            ModelName.lightgbm: {
                TaskName.classification: Pipeline([(ModelName.lightgbm.value,
                                                    LGBMClassifier(
                                                        categorical_features=self.categorical_features_indices,
                                                        verbose=-1, verbose_eval=-1))]),
                TaskName.regression: Pipeline([(ModelName.lightgbm.value,
                                                LGBMRegressor(categorical_features=self.categorical_features_indices,
                                                              verbose=-1,
                                                              verbose_eval=-1))])
            },
            ModelName.lightgbm_with_cat_encoder: {
                TaskName.classification: Pipeline(
                    [(ModelName.encoder.value, CatBoostEncoder(cols=self.categorical_features,
                                                               verbose=0)),
                     (ModelName.lightgbm.value, LGBMClassifier(verbosie=-1,
                                                               verbose_eval=-1))]),
                TaskName.regression: Pipeline([(ModelName.encoder.value, CatBoostEncoder(cols=self.categorical_features,
                                                                                         verbose=-1)),
                                               (ModelName.lightgbm.value, LGBMRegressor(verbose=-1,
                                                                                        verbose_eval=-1))]),
            },
            ModelName.xgboost: {
                TaskName.classification: Pipeline([(ModelName.xgboost.value, XGBClassifier())]),
                TaskName.regression: Pipeline([(ModelName.xgboost.value, XGBRegressor())])
            },
            ModelName.xgboost_with_cat_encoder: {
                TaskName.classification: Pipeline(
                    [(ModelName.encoder.value, CatBoostEncoder(cols=self.categorical_features,
                                                               verbose=0)),
                     (ModelName.xgboost.value, XGBClassifier())]),
                TaskName.regression: Pipeline(
                    [(ModelName.encoder.value, CatBoostEncoder(cols=self.categorical_features,
                                                               verbose=0)),
                     (ModelName.xgboost.value, XGBRegressor())]),
            }
        }
