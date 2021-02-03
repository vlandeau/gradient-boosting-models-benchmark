import pandas as pd
import numpy as np
from hamcrest import assert_that
from hamcrest.core.core import is_

from comparison.model_comparison import ModelComparison, TaskName

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
    model_comparison = ModelComparison(TaskName.regression, cross_validation_n_folds, features, numerical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for performance_and_training_time in comparison.values():
        assert_that(len(performance_and_training_time), is_(2))
        performance = performance_and_training_time[0]
        training_time = performance_and_training_time[1]
        assert_that(~np.isnan(performance))
        assert_that(~np.isnan(training_time))


def test_model_comparison_give_non_null_performance_and_categorical_feature():
    # Given
    features = pd.DataFrame({
        "string_feature": np.random.choice(["Paris", "London", "Madrid", "Roma"], n_samples),
        "numeric_feature": np.random.normal(size=n_samples)
    }, dtype="category")

    # When
    model_comparison = ModelComparison(TaskName.regression, cross_validation_n_folds, features, numerical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for performance_and_training_time in comparison.values():
        assert_that(len(performance_and_training_time), is_(2))
        performance = performance_and_training_time[0]
        training_time = performance_and_training_time[1]
        assert_that(~np.isnan(performance))
        assert_that(~np.isnan(training_time))


def test_model_comparison_give_non_null_performance_with_classification():
    # Given
    features = pd.DataFrame({
        "numeric_feature": np.random.normal(size=n_samples)
    })

    # When
    model_comparison = ModelComparison(TaskName.classification, cross_validation_n_folds, features, categorical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for performance_and_training_time in comparison.values():
        assert_that(len(performance_and_training_time), is_(2))
        performance = performance_and_training_time[0]
        training_time = performance_and_training_time[1]
        assert_that(~np.isnan(performance))
        assert_that(~np.isnan(training_time))


def test_model_comparison_give_non_null_performance_with_null_numerical_feature():
    # Given
    features = pd.DataFrame({
        "numeric_feature": list(np.random.normal(size=n_samples - 1)) + [None]
    })

    # When
    model_comparison = ModelComparison(TaskName.classification, cross_validation_n_folds, features, categorical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for performance_and_training_time in comparison.values():
        assert_that(len(performance_and_training_time), is_(2))
        performance = performance_and_training_time[0]
        training_time = performance_and_training_time[1]
        assert_that(~np.isnan(performance))
        assert_that(~np.isnan(training_time))


def test_model_comparison_give_non_null_performance_with_null_categorical_feature():
    # Given
    features = pd.DataFrame({
        "string_feature": list(np.random.choice(["Paris", "London", "Madrid", "Roma"], n_samples - 1)) + [None],
    }, dtype="category")

    # When
    model_comparison = ModelComparison(TaskName.regression, cross_validation_n_folds, features, numerical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for performance_and_training_time in comparison.values():
        assert_that(len(performance_and_training_time), is_(2))
        performance = performance_and_training_time[0]
        training_time = performance_and_training_time[1]
        assert_that(~np.isnan(performance))
        assert_that(~np.isnan(training_time))
