import pandas as pd
import numpy as np
from hamcrest import assert_that

from comparison.model_comparison import ModelComparison, TaskName, MODEL_SCORE

n_samples = 20
cross_validation_n_folds = 2
numerical_target = pd.Series(np.random.normal(size=n_samples))
categorical_target = pd.Series(np.random.choice(["Paris", "London", "Madrid", "Roma"], n_samples))


def test_model_comparison_give_non_null_performance_with_regression():
    # Given
    features = pd.DataFrame({
        "numeric_feature": np.random.normal(size=n_samples)
    })

    # When
    model_comparison = ModelComparison(TaskName.REGRESSION, cross_validation_n_folds, features, numerical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        performance = performance_and_training_time[MODEL_SCORE]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")


def test_model_comparison_give_non_null_performance_and_categorical_feature():
    # Given
    features = pd.DataFrame({
        "string_feature": np.random.choice(["Paris", "London", "Madrid", "Roma"], n_samples),
        "numeric_feature": np.random.normal(size=n_samples)
    }, dtype="category")

    # When
    model_comparison = ModelComparison(TaskName.REGRESSION, cross_validation_n_folds, features, numerical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        performance = performance_and_training_time[MODEL_SCORE]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")


def test_model_comparison_give_non_null_performance_with_classification():
    # Given
    features = pd.DataFrame({
        "numeric_feature": np.random.normal(size=n_samples)
    })

    # When
    model_comparison = ModelComparison(TaskName.CLASSIFICATION, cross_validation_n_folds, features, categorical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        performance = performance_and_training_time[MODEL_SCORE]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")


def test_model_comparison_give_non_null_performance_with_null_numerical_feature():
    # Given
    features = pd.DataFrame({
        "numeric_feature": list(np.random.normal(size=n_samples - 1)) + [None]
    })

    # When
    model_comparison = ModelComparison(TaskName.CLASSIFICATION, cross_validation_n_folds, features, categorical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        performance = performance_and_training_time[MODEL_SCORE]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")


def test_model_comparison_give_non_null_performance_with_null_categorical_feature():
    # Given
    features = pd.DataFrame({
        "string_feature": list(np.random.choice(["Paris", "London", "Madrid", "Roma"], n_samples - 1)) + [None],
    }, dtype="category")

    # When
    model_comparison = ModelComparison(TaskName.REGRESSION, cross_validation_n_folds, features, numerical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        performance = performance_and_training_time[MODEL_SCORE]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")
