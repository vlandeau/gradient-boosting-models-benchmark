from typing import Dict

from tqdm import tqdm

from comparison.comparison_datasets import get_comparison_datasets, ComparisonDataset, TaskName
from comparison.model_comparison import ModelName
from model_comparison import ModelComparison
from tuned_model_comparison import TunedModelComparison
import json


def get_comparison_default_models(dataset_infos: ComparisonDataset, dataset_name: str) \
        -> Dict[ModelName, Dict[str, object]]:
    print(f"Processing {dataset_name} dataset")
    comparison = ModelComparison(dataset_infos)
    return comparison.get_default_models_scores_and_training_time()


def get_comparison_tuned_models(dataset_infos: ComparisonDataset, dataset_name: str) \
        -> Dict[ModelName, Dict[str, object]]:
    print(f"Processing {dataset_name} dataset")
    comparison = TunedModelComparison(dataset_infos)
    return comparison.get_default_models_scores_and_training_time()


if __name__ == "__main__":
    comparison_datasets = get_comparison_datasets()

    perf_comparisons = {dataset_name: get_comparison_default_models(comparison_datasets[dataset_name], dataset_name)
                        for dataset_name in tqdm(comparison_datasets.keys(),
                                                 desc="Comparison of models with default hyperparameters")}
    with open("perf_comparison.json", "w") as default_performances_output_stream:
        json.dump(perf_comparisons, default_performances_output_stream)

    tuned_perf_comparisons = {dataset_name: get_comparison_tuned_models(comparison_datasets[dataset_name], dataset_name)
                              for dataset_name in tqdm(comparison_datasets.keys(),
                                                       desc="Comparison of models with tuned hyperparameters")}
    with open("tuned_perf_comparison.json", "w") as tuned_performances_output_stream:
        json.dump(tuned_perf_comparisons, tuned_performances_output_stream)
