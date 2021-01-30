import time
from enum import Enum
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


class TaskNames(Enum):
    classification = "classification_task"
    regression = "regression_task"


class ModelNames(Enum):
    catboost = "catboost"
    xgboost = "xgboost"
    lightgbm = "lightgbm"


MODELS_TO_COMPARE: Dict[ModelNames, Dict[TaskNames, callable]] = {
    ModelNames.catboost: {
        TaskNames.classification: CatBoostClassifier,
        TaskNames.regression: CatBoostRegressor
    },
    ModelNames.lightgbm: {
        TaskNames.classification: LGBMClassifier,
        TaskNames.regression: LGBMRegressor
    },
    ModelNames.xgboost: {
        TaskNames.classification: XGBClassifier,
        TaskNames.regression: XGBRegressor
    }
}


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
        categorical_features = set(features.columns) - numeric_features
        self.categorical_features_indices = list(np.where(features.columns.isin(categorical_features))[0])

        encoded_categorical_features = {
            categorical_feature:
                LabelEncoder().fit_transform(features[categorical_feature].astype(str).fillna(self.unknown_category))
            for categorical_feature in categorical_features
        }
        self.features = features.assign(**encoded_categorical_features) \
            .fillna({numeric_feature: self.unknown_numeric_value
                     for numeric_feature in numeric_features})
        self.target = target

    def get_default_models_scores_and_training_time(self) -> Dict:
        return {model_name: self._get_default_model_score_and_training_time(model_name)
                for model_name in MODELS_TO_COMPARE.keys()}

    def _get_default_model_score_and_training_time(self, model_name: ModelNames) -> Tuple[float, float]:
        model_class = MODELS_TO_COMPARE[model_name][self.task_name]
        if model_name == ModelNames.catboost:
            model = model_class(cat_features=self.categorical_features_indices)
        elif model_name == ModelNames.lightgbm:
            model = model_class(categorical_features=self.categorical_features_indices)
        else:
            model = model_class()

        start_time = time.time()
        cross_val_scores = cross_val_score(model,
                                           self.features,
                                           self.target,
                                           cv=self.cross_validation_n_folds,
                                           n_jobs=-1)
        end_time = time.time()
        return np.mean(cross_val_scores), end_time - start_time
