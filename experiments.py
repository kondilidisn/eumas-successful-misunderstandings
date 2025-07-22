from experiment_utils import *

def population_increase_experiment_wrapper(cfg: DictConfig) -> dict[int, dict[str, Union[int, float]]]:

    high_level_experiment_statistics_dicts: dict = dict()
    high_level_experiment_statistics_dicts["Initial Population"] = dict()
    high_level_experiment_statistics_dicts["Final Population"] = dict()


    if cfg.wandb.use and cfg.wandb.record_individual_experiments:
        init_wandb(cfg)


    reward_table = get_reward_table(cfg)

    old_population:list[Agent] = []

    for _ in range(cfg.experiment.init_population):
        old_population.append(Agent(cfg,
            interpretation_dimensions_per_role=
                [len(reward_table),len(reward_table[0])], population_size=cfg.experiment.init_population))


    high_level_experiment_statistics_dicts["Initial Population"] = run_stage_experiment(cfg, agents=old_population, reward_table=reward_table, start_wandb=False)

    new_population:list[Agent] = []

    for _ in range(cfg.experiment.final_population - cfg.experiment.init_population):
        new_population.append(Agent(cfg, interpretation_dimensions_per_role=[len(reward_table), len(reward_table[0])], population_size=cfg.experiment.final_population))

    final_population: list[Agent] = old_population + new_population

    high_level_experiment_statistics_dicts["Final Population"] = run_stage_experiment(cfg, agents=final_population, reward_table=reward_table, start_wandb=False)


    if cfg.wandb.use and cfg.wandb.record_individual_experiments:
        for steps in high_level_experiment_statistics_dicts["Initial Population"]:
            data_dict_to_register = {}

            for stage in high_level_experiment_statistics_dicts:
                stage_data = dict()
                for metric in high_level_experiment_statistics_dicts[stage][steps]:

                    stage_data[stage + "/" + metric] = high_level_experiment_statistics_dicts[stage][steps][metric]
                data_dict_to_register |= stage_data


            wandb.log(data=data_dict_to_register, step=steps)

        wandb.finish()

    return high_level_experiment_statistics_dicts


def agents_initially_grouped_experiment_wrapper(cfg: DictConfig) -> dict[int, dict[str, Union[int, float]]]:


    high_level_experiment_statistics_dicts: dict = dict()
    high_level_experiment_statistics_dicts["Init Grouped Interactions"] = dict()
    high_level_experiment_statistics_dicts["Final Random Interactions"] = dict()


    cfg.experiment.agent_pairing = "two_teams"

    if cfg.wandb.use and cfg.wandb.record_individual_experiments:
        init_wandb(cfg)

    stag_hunt_rewards = get_reward_table(cfg)

    agents: list[Agent] = []

    for _ in range(cfg.experiment.population_size):
        agents.append(Agent(cfg,
                                    interpretation_dimensions_per_role=
                                    [len(stag_hunt_rewards), len(stag_hunt_rewards[0])]))

    high_level_experiment_statistics_dicts["Init Grouped Interactions"] = run_stage_experiment(cfg, agents=agents, reward_table=stag_hunt_rewards, start_wandb=False)

    cfg.experiment.agent_pairing = "random"

    high_level_experiment_statistics_dicts["Final Random Interactions"] = run_stage_experiment(cfg, agents=agents, reward_table=stag_hunt_rewards, start_wandb=False)

    if cfg.wandb.use and cfg.wandb.record_individual_experiments:
        for steps in high_level_experiment_statistics_dicts["Init Grouped Interactions"]:
            data_dict_to_register = {}
            for stage in high_level_experiment_statistics_dicts:
                stage_data = {stage + "/" + metric: high_level_experiment_statistics_dicts[stage][steps][metric] for metric in high_level_experiment_statistics_dicts[stage][steps]}
                data_dict_to_register |= stage_data

            wandb.log(data=data_dict_to_register, step=steps)

        wandb.finish()


    return high_level_experiment_statistics_dicts



def run_stage_experiment(cfg: DictConfig, agents:Optional[list[Agent]]=None, experiment_description:str= "", start_wandb:bool=True, reward_table:Optional[list]=None) -> dict[int, dict[str, Union[int, float]]]:


    if cfg.wandb.use and cfg.wandb.record_individual_experiments and start_wandb:
        init_wandb(cfg, group=experiment_description)

    if reward_table is None:
        reward_table = get_reward_table(cfg=cfg)

    if agents is None:
        agents:list[Agent] = []

        for _ in range(cfg.experiment.population_size):
            agents.append(Agent(cfg,
                                interpretation_dimensions_per_role=
                    [len(reward_table), len(reward_table[0])]))


    rewards: list[float] = []
    speaker_rewards: list[float] = []
    listener_rewards: list[float] = []

    speakers_intent_met_ratios: list[float] = []
    successful_misunderstandings: list[float] = []

    experiment_statistics_on_episode_level:dict[int, dict[str, Union[int, float]]] = dict()

    for episode_counter in range(1, cfg.experiment.episodes + 1):

        if cfg.experiment.agent_pairing == "random":
            speaker_index, listener_index = random.sample(range(len(agents)), 2)
        elif cfg.experiment.agent_pairing == "two_teams":

            even_agent_indices = [i for i in range(0, len(agents), 2)]
            odd_agent_indices = [i for i in range(1, len(agents), 2)]

            speaker_is_even = random.choice([True, False])

            if speaker_is_even:
                speaker_candidate_indices = even_agent_indices
                listener_candidate_indices = odd_agent_indices
            else:
                listener_candidate_indices = even_agent_indices
                speaker_candidate_indices = odd_agent_indices

            speaker_index: int = random.choice(speaker_candidate_indices)
            listener_index:int = random.choice(listener_candidate_indices)

        else:
            raise ValueError("'cfg.experiment.agent_pairing' not recognised: " + cfg.experiment.agent_pairing)

        speaker_agent = agents[speaker_index]
        listener_agent = agents[listener_index]

        environment_state_observation:int = random.randint(0, len(reward_table) - 1)

        communicated_word:str = speaker_agent.select_word_to_communicate_give_role_and_observation(role=0,
                                                                                                    observed_dim=environment_state_observation)

        listener_action: int = listener_agent.select_dimension_given_role_and_word(role=1, word=communicated_word)


        speakers_intent = speaker_agent.select_dimension_given_role_and_word(role=1, word=communicated_word, dont_update_epsilon=True)
        listeners_hypothesis_of_speakers_observation = listener_agent.select_dimension_given_role_and_word(role=0, word=communicated_word, dont_update_epsilon=True)


        speakers_intent_met = float(listener_action == speakers_intent)

        speakers_reward:int = reward_table[environment_state_observation][listener_action][0]
        listeners_reward:int = reward_table[environment_state_observation][listener_action][1]


        speaker_agent.update_word_interpretation_for_role_group(role=0, observed_dim=environment_state_observation,
                                                                    word=communicated_word, reward=speakers_reward)

        listener_agent.update_word_interpretation_for_role_group(role=1, observed_dim=listener_action,
                                                                 word=communicated_word, reward=listeners_reward)


        if cfg.experiment.print_debug_episode_summary:
            print(f"Sp.Act.={environment_state_observation}, Sp.words={speaker_agent.word_interpretations} Sp_Word_Ind={speaker_agent.words.index(communicated_word)}, Li.words={listener_agent.word_interpretations}, Li_Word_Ind={listener_agent.words.index(communicated_word)}, Act={listener_action}")
            print(f"Sp.word={speaker_agent.word_interpretations[speaker_agent.words.index(communicated_word)]}, Li.word={listener_agent.word_interpretations[listener_agent.words.index(communicated_word)]}, , Sp_R={speakers_reward}, Li_R={listeners_reward}")



        speaker_agent.reset_words_used_counter_and_forget_words([communicated_word])
        listener_agent.reset_words_used_counter_and_forget_words([communicated_word])

        speaker_agent.update_word_used(communicated_word)
        listener_agent.update_word_used(communicated_word)



        rewards.append((speakers_reward + listeners_reward)/2)

        speaker_agent.update_speakers_action_expected_reward(selected_action=environment_state_observation, reward=speakers_reward)

        speakers_intent_met_ratios.append(speakers_intent_met)


        successful_misunderstanding = (speakers_reward == listeners_reward) and (speakers_intent_met == 0)

        successful_misunderstandings.append(successful_misunderstanding)


        speaker_rewards.append(speakers_reward)
        listener_rewards.append(listeners_reward)

        speaker_agent.add_reward_per_role_memory(role=0, reward=speakers_reward)
        listener_agent.add_reward_per_role_memory(role=1, reward=listeners_reward)


        if communicated_word is not None:

            if cfg.experiment.print_episode_summary:
                print(
                    f"Episode {episode_counter:6}:  Agents: ({speaker_index}-{listener_index}) Sp_Act={environment_state_observation} Sp_H={speakers_intent}, (Word {communicated_word}) inter.={speaker_agent.word_interpretations[speaker_agent.words.index(communicated_word)]}, " + \
                    f"H_Obs={listeners_hypothesis_of_speakers_observation}, Act={listener_action}, S_Word_inter={listener_agent.word_interpretations[listener_agent.words.index(communicated_word)]}, " + \
                    f"Rewards:{speakers_reward}|{listeners_reward}")

            if cfg.experiment.print_debug_episode_summary:
                print(f", Sp_R={speakers_reward}, Li_R={listeners_reward}, Sp_Upd_word={speaker_agent.word_interpretations[speaker_agent.words.index(communicated_word)]}, Li_Upd_word={listener_agent.word_interpretations[listener_agent.words.index(communicated_word)]}")

        else:

            if cfg.experiment.print_episode_summary:
                print(
                    f"Episode {episode_counter:6}: S_Id={speaker_index}, Sp_Act={environment_state_observation}, " + \
                    f"H_Act={speakers_intent}, L_Id={listener_index}, " + \
                    f"Act={listener_action}, , Sp_R={speakers_reward}, Li_R={listeners_reward}, Succ_Miss={successful_misunderstanding}")

            if cfg.experiment.print_debug_episode_summary:
                print(f", Sp_R={speakers_reward}, Li_R={listeners_reward}")


        if episode_counter % cfg.experiment.report_every_episodes == 0:
            experiment_stats = calculate_experiment_stats(episode=episode_counter, agents=agents,
                                                          speaker_rewards=speaker_rewards, listener_rewards=listener_rewards,
                                                          successful_misunderstandings=successful_misunderstandings,
                                                          speakers_intent_met_ratios=speakers_intent_met_ratios, cfg=cfg)

            experiment_statistics_on_episode_level[episode_counter] = experiment_stats

            if cfg.wandb.use and cfg.wandb.record_individual_experiments and start_wandb:
                if experiment_description != "":
                    wandb.log(data={experiment_description + "/" + exp_rec: experiment_stats[exp_rec] for exp_rec in experiment_stats}, step=episode_counter)
                else:
                    wandb.log(data= experiment_stats, step=episode_counter)

    # if we have not reported the experiment statistics for the last episode already, we report them
    if cfg.experiment.episodes not in experiment_statistics_on_episode_level:
        experiment_stats = calculate_experiment_stats(episode=cfg.experiment.episodes, agents=agents,
                                                      speaker_rewards=speaker_rewards, listener_rewards=listener_rewards,
                                                      successful_misunderstandings=successful_misunderstandings,
                                                      speakers_intent_met_ratios=speakers_intent_met_ratios, cfg=cfg)

        experiment_statistics_on_episode_level[cfg.experiment.episodes] = experiment_stats

        if cfg.wandb.use and cfg.wandb.record_individual_experiments and start_wandb:
            if experiment_description != "":
                wandb.log(data={experiment_description + "/" + exp_rec: experiment_stats[exp_rec] for exp_rec in experiment_stats}, step=cfg.experiment.episodes)
            else:
                wandb.log(data=experiment_stats, step=cfg.experiment.episodes)

    if start_wandb and cfg.wandb.use and cfg.wandb.record_individual_experiments:
        wandb.finish()

    return experiment_statistics_on_episode_level


