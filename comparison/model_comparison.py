import time
from enum import Enum
from multiprocessing import Pool
from typing import Dict

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

DEFAULT_PARAMETERS = "default_parameters"

TRAINING_TIME = "training_time"

MODEL_SCORE = "model_score"


class TaskName(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelName(Enum):
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LIGHTGBM_WITH_CATBOOST_ENCODER = "lightgbm_with_catboost_encoder"
    CATBOOST_ENCODER = "catboost_encoder"


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

    def get_default_models_scores_and_training_time(self) -> Dict[ModelName, Dict[str, object]]:
        with Pool(processes=4) as pool:
            results = pool.map(self._get_default_model_score_and_training_time, self.models_to_compare.keys())
        return {model_name: performance_and_time for model_name, performance_and_time
                in zip(self.models_to_compare.keys(), results)}

    def _get_default_model_score_and_training_time(self, model_name: ModelName) -> Dict[str, object]:
        model = self.models_to_compare[model_name][self.task_name]

        start_time = time.time()
        cross_val_scores = cross_val_score(model,
                                           self.features,
                                           self.target,
                                           cv=KFold(self.cross_validation_n_folds,
                                                    shuffle=True),
                                           n_jobs=-1)
        end_time = time.time()
        return {MODEL_SCORE: np.mean(cross_val_scores),
                TRAINING_TIME: end_time - start_time,
                DEFAULT_PARAMETERS: model.get_params(deep=True)}

    @property
    def models_to_compare(self) -> Dict[ModelName, Dict[TaskName, Pipeline]]:
        return {
            ModelName.CATBOOST: {
                TaskName.CLASSIFICATION: Pipeline([(ModelName.CATBOOST.value,
                                                    CatBoostClassifier(cat_features=self.categorical_features_indices,
                                                                       verbose=0))]),
                TaskName.REGRESSION: Pipeline([(ModelName.CATBOOST.value,
                                                CatBoostRegressor(cat_features=self.categorical_features_indices,
                                                                  verbose=0))])
            },
            ModelName.LIGHTGBM: {
                TaskName.CLASSIFICATION: Pipeline([(ModelName.LIGHTGBM.value,
                                                    LGBMClassifier(
                                                        categorical_features=self.categorical_features_indices,
                                                        verbose=-1, verbose_eval=-1))]),
                TaskName.REGRESSION: Pipeline([(ModelName.LIGHTGBM.value,
                                                LGBMRegressor(categorical_features=self.categorical_features_indices,
                                                              verbose=-1,
                                                              verbose_eval=-1))])
            },
            ModelName.LIGHTGBM_WITH_CATBOOST_ENCODER: {
                TaskName.CLASSIFICATION: Pipeline(
                    [(ModelName.CATBOOST_ENCODER.value, CatBoostEncoder(cols=self.categorical_features,
                                                                        verbose=0)),
                     (ModelName.LIGHTGBM.value, LGBMClassifier(verbosie=-1,
                                                               verbose_eval=-1))]),
                TaskName.REGRESSION: Pipeline([(ModelName.CATBOOST_ENCODER.value, CatBoostEncoder(cols=self.categorical_features,
                                                                                                  verbose=-1)),
                                               (ModelName.LIGHTGBM.value, LGBMRegressor(verbose=-1,
                                                                                        verbose_eval=-1))]),
            },
            ModelName.XGBOOST: {
                TaskName.CLASSIFICATION: Pipeline([(ModelName.XGBOOST.value, XGBClassifier())]),
                TaskName.REGRESSION: Pipeline([(ModelName.XGBOOST.value, XGBRegressor())])
            }
        }
