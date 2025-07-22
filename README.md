# This Repository contains the code and instructions needed to reproduce the experiments presented on the manuscript:

    "Successful Misunderstandings: Learning to Coordinate Without Being Understood", 
    Nikolaos Kondylidis, Anil Yaman, Frank van Harmelen, Erman Acar, Annette Ten Teije, 
    Presented at the 22nd European Conference on Multi-Agent Systems (EUMAS 2025), Bucharest

# Experiment Reproducibility


### 1 Download or clone this git


### 2 Create python environment and install required libraries


    python3 -m venv suc_mis_venv
    source suc_mis_venv/bin/activate
    pip install -r requirements.txt 

### (change "experiment.num_workers=10" to set number of threads to work in parallel)


### Experiment 1:
    python main.py experiment.init_population=2 experiment.final_population=3 experiment.reward_table=random_simple experiment.experiment_setting=population_increase_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5 experiment.num_workers=10


### Experiment 2:
    python main.py experiment.init_population=3 experiment.final_population=4 experiment.reward_table=random_simple experiment.experiment_setting=population_increase_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5 experiment.num_workers=10

### Experiment 3:
    python main.py experiment.population_size=3 experiment.reward_table=random_simple experiment.experiment_setting=agents_initially_grouped_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5 experiment.num_workers=10


### Experiments for Validating with more complicated Reward functions:



#### Non-symmeic reward function:

    % Experiment 1
    python main.py experiment.init_population=2 experiment.final_population=3 experiment.reward_table=random_simple_non_symmetric experiment.experiment_setting=population_increase_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5
    % Experiment 2
    python main.py experiment.init_population=3 experiment.final_population=4 experiment.reward_table=random_simple_non_symmetric experiment.experiment_setting=population_increase_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5
    % Experiment 3
    python main.py experiment.population_size=3 experiment.reward_table=random_simple_non_symmetric experiment.experiment_setting=agents_initially_grouped_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5

#### 3x3 Reward table, 3 States & 3 Actions:

    % Experiment 1
    python main.py experiment.init_population=2 experiment.final_population=3 experiment.reward_table=random_3x3 experiment.experiment_setting=population_increase_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5
    % Experiment 2
    python main.py experiment.init_population=3 experiment.final_population=4 experiment.reward_table=random_3x3 experiment.experiment_setting=population_increase_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5
    % Experiment 3
    python main.py experiment.population_size=3 experiment.reward_table=random_3x3 experiment.experiment_setting=agents_initially_grouped_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5


#### Experiment 3 with 4 agents:
    python main.py experiment.population_size=4 experiment.reward_table=random_simple experiment.experiment_setting=agents_initially_grouped_experiment experiment.episodes=10000 experiment.repetitions=1000 agent.apply_epsilon_in_episode_ratio=0.5


