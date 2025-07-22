from typing import Optional, Union, Tuple
import numpy.typing as npt
import numpy as np
import random
import statistics

from omegaconf import DictConfig, OmegaConf

import wandb


def apply_softmax_per_row(array: npt.NDArray) -> npt.NDArray:
    if len(array.shape) > 1:
        max_values = np.max(array, axis=1, keepdims=True)
        numerator = np.exp(array - max_values)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        softmaxed_array = numerator / denominator
    else:
        max_values = np.max(array)
        numerator = np.exp(array - max_values*np.ones_like(array))
        denominator = np.sum(numerator)
        if denominator != 0:
            softmaxed_array = numerator / denominator
        else:
            softmaxed_array = array
    return softmaxed_array

def mean_cosine_similarity(vectors):
    vectors += 0.1
    if vectors.shape[0] == 0:
        raise ValueError("Empty vectors")
    centroid = np.mean(vectors, axis=0)

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    similarities = [cosine_similarity(vec, centroid) for vec in vectors]

    if len(similarities) == 0:
        raise ValueError("Empty similarities")
    average_similarity = np.mean(similarities)

    return float(average_similarity)


def add_results_dictionary_values_to_overall_results_dictionary(total_evaluation_metrics_per_evaluation_step:dict[int, dict[str, list[float]]], evaluation_metrics_per_step:dict[int, dict[str, float]]) -> dict[int, dict[str, list[float]]]:
    for evaluation_step in evaluation_metrics_per_step:
        # initialize the results' dictionary if necessary
        if evaluation_step not in total_evaluation_metrics_per_evaluation_step:
            total_evaluation_metrics_per_evaluation_step[evaluation_step] = {}
        for evaluation_metric in evaluation_metrics_per_step[evaluation_step]:
            # initialize the results' list if necessary
            if evaluation_metric not in total_evaluation_metrics_per_evaluation_step[evaluation_step]:
                total_evaluation_metrics_per_evaluation_step[evaluation_step][evaluation_metric] = []
            # add the result of this step, for the evaluation metric, in the corresponding list
            total_evaluation_metrics_per_evaluation_step[evaluation_step][evaluation_metric].append(
                evaluation_metrics_per_step[evaluation_step][evaluation_metric])
    return total_evaluation_metrics_per_evaluation_step

def aggregate_experiment_repetition_results(total_evaluation_metrics_per_evaluation_step:dict[int, dict[str, list[float]]], metrics_histogram_bin_edges:Optional[list[float]]) -> Union[tuple[dict[int, dict[str, float]], dict[int, dict[str, float]], dict[int, dict[str, float]]], Tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]]:
    mean_eval_metrics_pers_step: dict[int, dict[str, float]] = {}
    std_eval_metrics_pers_step: dict[int, dict[str, float]] = {}

    histogram_eval_metrics_pers_step: dict[int, dict[str, float]] = {}

    for eval_step in total_evaluation_metrics_per_evaluation_step:
        if eval_step not in mean_eval_metrics_pers_step:
            mean_eval_metrics_pers_step[eval_step] = dict()
            std_eval_metrics_pers_step[eval_step] = dict()
            histogram_eval_metrics_pers_step[eval_step] = dict()
        for eval_metric in total_evaluation_metrics_per_evaluation_step[eval_step]:
            recorded_evaluation_metrics_values = total_evaluation_metrics_per_evaluation_step[eval_step][eval_metric]
            mean_eval_metrics_pers_step[eval_step][eval_metric] = sum(recorded_evaluation_metrics_values) / len(recorded_evaluation_metrics_values)
            if len(total_evaluation_metrics_per_evaluation_step[eval_step][eval_metric]) == 1:
                std_eval_metrics_pers_step[eval_step][eval_metric] = 0.0
            else:
                std_eval_metrics_pers_step[eval_step][eval_metric] = statistics.stdev(recorded_evaluation_metrics_values)

            if metrics_histogram_bin_edges is not None and ("Average Reward" in eval_metric or "Average Semantic Alignment Score" in eval_metric or "Speaker's Intent Met Ratio" in eval_metric or "Successful Misunderstanding Ratio" in eval_metric):
                hist, bin_edges = np.histogram(recorded_evaluation_metrics_values, bins=metrics_histogram_bin_edges)
                histogram_eval_metrics_pers_step[eval_step][eval_metric] = hist.tolist()

    if metrics_histogram_bin_edges is not None:
        return mean_eval_metrics_pers_step, std_eval_metrics_pers_step, histogram_eval_metrics_pers_step
    else:
        return mean_eval_metrics_pers_step, std_eval_metrics_pers_step


def init_wandb(cfg: DictConfig, group:str="", mean_run:bool=False):
    # start a new wandb run to track this repetition

    config_dict = {
            "Population Size": cfg.experiment.population_size,

            "Repetitions": cfg.experiment.repetitions,

            "Run Name": cfg.wandb.run_name,

            "Mean Run": mean_run,

            "Reward table": cfg.experiment.stag_hunt.reward_table,

            "Agent use epsilon": cfg.agent.use_epsilon_greedy,
            "Agent eps start": cfg.agent.epsilon_start,
            "Agent eps min": cfg.agent.epsilon_min,
            "Agent eps ep ratio": cfg.agent.apply_epsilon_in_episode_ratio,
            "Agent lear rate": cfg.agent.learning_rate,
            "Agent word forget policy": cfg.agent.forget_oldest_word_or_least_useful,
            "Agent _vocab_size": cfg.agent.max_vocab_size,
            "Agent word forget thr": cfg.agent.forget_word_threshold,

            "R_Window_Size": cfg.experiment.average_window_size,
            "Word_Length": cfg.experiment.word_length,

            "Init P. Size": cfg.experiment.init_population,
            "Final P. Size": cfg.experiment.final_population,

            "experiment_setting": cfg.experiment.experiment_setting
        }




    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        # name=str(time.time()),
        group=group,
        # track hyperparameters and run metadata
        config=config_dict
    )

def register_experiment_stats_to_wandb(high_level_experiment_statistics_dicts):
    random_stage = list(high_level_experiment_statistics_dicts)[0]

    for steps in high_level_experiment_statistics_dicts[random_stage]:
        data_dict_to_register = {}
        for stage in high_level_experiment_statistics_dicts:
            stage_data = {stage + "/" + metric: high_level_experiment_statistics_dicts[stage][steps][metric] for
                          metric in high_level_experiment_statistics_dicts[stage][steps]}
            data_dict_to_register |= stage_data

        wandb.log(data=data_dict_to_register, step=steps)
