
from experiments import *

from tqdm.contrib.concurrent import process_map

import hydra


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Optional[DictConfig] = None) -> None:

    experiment_stats_from_all_executions:dict = dict()


    if cfg.experiment.average_window_size == 0:
        cfg.experiment.average_window_size = cfg.experiment.report_every_episodes

    if cfg.wandb.use:
        # set experiment logger config
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.login(key=cfg.wandb.key, relogin=True)

    if cfg.experiment.experiment_setting == "population_increase_experiment":
        experiment_function = population_increase_experiment_wrapper
    elif cfg.experiment.experiment_setting == "agents_initially_grouped_experiment":
        experiment_function = agents_initially_grouped_experiment_wrapper

    else:
        raise ValueError(f"cfg.experiment.experiment_setting {cfg.experiment.experiment_setting}, not recognised.")

    experiment_stats_from_all_stages_list = process_map(experiment_function,
                                                          [cfg for _ in range(cfg.experiment.repetitions)],
                                                          max_workers=cfg.experiment.num_workers)

    for experiment_stats_from_all_stages in experiment_stats_from_all_stages_list:

        for experiment_stage in experiment_stats_from_all_stages:

            if experiment_stage not in experiment_stats_from_all_executions:
                experiment_stats_from_all_executions[experiment_stage] = dict()
            add_results_dictionary_values_to_overall_results_dictionary(experiment_stats_from_all_executions[experiment_stage], experiment_stats_from_all_stages[experiment_stage])

    mean_experiment_records = dict()

    for experiment_stage in experiment_stats_from_all_executions:
        print(experiment_stage)
        mean_experiment_stats, std_experiment_stats, histogram_eval_metrics_pers_step = aggregate_experiment_repetition_results(experiment_stats_from_all_executions[experiment_stage], metrics_histogram_bin_edges=cfg.experiment.metrics_histogram_bin_edges)
        mean_experiment_records[experiment_stage] = mean_experiment_stats
        print_aggregated_experiment_stats(mean_experiment_stats, std_experiment_stats, histogram_eval_metrics_pers_step, cfg.experiment.repetitions)


    if cfg.wandb.use and not cfg.wandb.record_individual_experiments:
        init_wandb(cfg, mean_run=True)
        register_experiment_stats_to_wandb(mean_experiment_records)

        wandb.finish()

if __name__ == '__main__':
    main()
