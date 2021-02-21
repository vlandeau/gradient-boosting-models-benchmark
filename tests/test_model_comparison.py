import pandas as pd
import numpy as np
from hamcrest import assert_that, is_

from comparison.comparison_datasets import ComparisonDataset, TaskName
from comparison.model_comparison import ModelComparison, MODEL_SCORE


n_samples = 20
cross_validation_n_folds = 2
numerical_target = pd.Series(np.random.normal(size=n_samples))
categorical_target = pd.Series(np.random.choice(["Paris", "London", "Madrid", "Roma"], n_samples))


def test_model_comparison_give_non_null_performance_with_regression():
    # Given
    features = pd.DataFrame({
        "numeric_feature": np.random.normal(size=n_samples)
    })
    comparison_dataset = ComparisonDataset(TaskName.REGRESSION, features, numerical_target, cross_validation_n_folds)

    # When
    comparison = ModelComparison(comparison_dataset).get_models_scores_and_training_time()

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
    comparison_dataset = ComparisonDataset(TaskName.REGRESSION, features, categorical_target, cross_validation_n_folds)

    # When
    comparison = ModelComparison(comparison_dataset).get_models_scores_and_training_time()

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
    comparison_dataset = ComparisonDataset(TaskName.CLASSIFICATION, features, categorical_target,
                                           cross_validation_n_folds)

    # When
    comparison = ModelComparison(comparison_dataset).get_models_scores_and_training_time()

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
    comparison_dataset = ComparisonDataset(TaskName.CLASSIFICATION, features, categorical_target,
                                           cross_validation_n_folds)

    # When
    comparison = ModelComparison(comparison_dataset).get_models_scores_and_training_time()

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
    comparison_dataset = ComparisonDataset(TaskName.REGRESSION, features, categorical_target, cross_validation_n_folds)

    # When
    comparison = ModelComparison(comparison_dataset).get_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        performance = performance_and_training_time[MODEL_SCORE]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")


def test_encode_datetime_columns_as_int():
    # Given
    date_column = "date_column"
    other_string_column = "other_string_column"
    other_numeric_column = "other_numeric_column"

    df = pd.DataFrame({date_column: ['2017-02-04 18:41:00'],
                       other_numeric_column: [1],
                       other_string_column: ["something"]})

    # When
    parsed_df = ModelComparison._encode_date_columns_as_int(df, 1)

    # Then
    parsed_df_dtypes = parsed_df.dtypes
    df_dtypes = df.dtypes

    assert_that(parsed_df_dtypes[date_column], is_("float"))
    assert_that(parsed_df_dtypes[other_string_column], is_(df_dtypes[other_string_column]))
    assert_that(parsed_df_dtypes[other_numeric_column], is_(df_dtypes[other_numeric_column]))
