from agents import Agent

from collections import Counter
from utils import *

def print_aggregated_experiment_stats(mean_experiment_stats:dict, std_experiment_stats:dict, histogram_eval_metrics_pers_step:dict, repetitions:int, prefix:str=""):
    print(f"{prefix} | Aggregated experiment results over {repetitions} repetitions:")
    for steps in mean_experiment_stats:
        print_str = f"{steps:7.0f}"
        for metric in mean_experiment_stats[steps]:
            if steps in histogram_eval_metrics_pers_step and metric in histogram_eval_metrics_pers_step[steps]:
                print_str += f", {metric}: {mean_experiment_stats[steps][metric]:5.2f} ({std_experiment_stats[steps][metric]:4.2f}) {histogram_eval_metrics_pers_step[steps][metric]}"
            else:
                print_str += f", {metric}: {mean_experiment_stats[steps][metric]:5.2f} ({std_experiment_stats[steps][metric]:4.2f})"
        print(print_str)


def calculate_population_vocabulary_interpretation(agents: list[Agent], print_messages:bool): # , print_agents_env_understanding:bool=True):

    all_remaining_words: set[str] = set()

    for agent in agents:
        all_remaining_words.update(agent.words)

    number_of_words: int = len(all_remaining_words)

    complete_interpretation_agreement_score: float = 0


    number_of_roles: int = len(agents[0].interpretation_dimensions_per_role)


    for word in all_remaining_words:
        word_interpretations = []
        word_average_return_per_role = []
        word_used_counters = []
        for agent in agents:
            agents_word_average_return_per_role = []
            if word in agent.words:
                index_of_word = agent.words.index(word)
                word_interpretations.append(agent.word_interpretations[index_of_word, :].tolist())
                for role in range(number_of_roles):
                    agents_word_average_return_per_role.append(agent.get_average_score_of_word_per_role(word, role))

                word_average_return_per_role.append(agents_word_average_return_per_role)

                word_used_counters.append(agent.word_used_counter[agent.words.index(word)])

        np_word_interpretations = np.array(word_interpretations)


        word_agreement_accuracy: float = 0

        best_interpretation_per_word_per_role: npt.NDArray = np.zeros((np_word_interpretations.shape[0], number_of_roles))

        for role in range(number_of_roles):
            np_word_interpretations[:, agents[0].interpret_dims_indices_per_role[role]: agents[0].interpret_dims_indices_per_role[role + 1]] = \
                apply_softmax_per_row(np_word_interpretations[:, agents[0].interpret_dims_indices_per_role[role]: agents[0].interpret_dims_indices_per_role[role + 1]])

            best_interpretation_per_word_per_role[:, role] = np.argmax(np_word_interpretations[:, agents[0].interpret_dims_indices_per_role[role]: agents[0].interpret_dims_indices_per_role[role + 1]], axis=1)

        for role in range(number_of_roles):
            word_role_interpretations = np_word_interpretations[:,
                                        agents[0].interpret_dims_indices_per_role[role]:
                                        agents[0].interpret_dims_indices_per_role[role + 1]]

            interpreted_dimensions = np.argmax(word_role_interpretations, axis=1).tolist()

            interpreted_dimensions_counter:Counter = Counter(interpreted_dimensions)

            frequency_of_most_common_interpretation:int = interpreted_dimensions_counter.most_common()[0][1]

            word_role_agreement_accuracy = (frequency_of_most_common_interpretation - 1) / (len(agents) - 1)

            word_agreement_accuracy += word_role_agreement_accuracy

        word_agreement_accuracy /= number_of_roles

        complete_interpretation_agreement_score += word_agreement_accuracy


        if print_messages:

            word_interpretation_print_statement = f"{word} (Semantic coherence: {round(word_agreement_accuracy, 2)}. Pref_Dim|Av_Rew: ["

            for agent_index in range(len(word_average_return_per_role)):
                word_interpretation_print_statement += "["
                for role in range(len(word_average_return_per_role[agent_index])):
                    word_interpretation_print_statement += str(int(best_interpretation_per_word_per_role[agent_index][role])) + ":" + str(round(word_average_return_per_role[agent_index][role],1))
                    if role < len(word_average_return_per_role[agent_index]) - 1:
                        word_interpretation_print_statement += ", "
                    else:
                        if agent_index < len(word_average_return_per_role) -1:
                            word_interpretation_print_statement += "], "
                        else:
                            word_interpretation_print_statement += "]], "

            word_interpretation_print_statement += "| Details: "
            for i, row in enumerate(word_interpretations):
                word_interpretation_print_statement += str([f"{x:.1f}" for x in row]) + f" ({int(word_used_counters[i])})"

                if i != len(word_interpretations) - 1:
                    word_interpretation_print_statement += ", "

            print(word_interpretation_print_statement)

    complete_interpretation_agreement_score /= number_of_words
    if print_messages:
        print("Overall interpretation agreement score:", f"{complete_interpretation_agreement_score:.2f}")

    population_word_interpretation_alignment_dict: dict = {"Average Number of Words per Agent": number_of_words,
                                       "Average Semantic Alignment Score": complete_interpretation_agreement_score}

    return population_word_interpretation_alignment_dict


def calculate_experiment_stats(episode:int, agents: list[Agent], speaker_rewards: list[float], listener_rewards: list[float], successful_misunderstandings :list[float], speakers_intent_met_ratios: list[float], cfg: DictConfig) -> dict[str, Union[int, float]]:

    speaker_reward_moving_average = np.mean(speaker_rewards[-cfg.experiment.average_window_size:])
    listener_reward_moving_average = np.mean(listener_rewards[-cfg.experiment.average_window_size:])
    succ_misunderstandings_moving_average = np.mean(successful_misunderstandings[-cfg.experiment.average_window_size:])
    speakers_intent_met_moving_average = np.mean(speakers_intent_met_ratios[-cfg.experiment.average_window_size:])

    average_reward_moving_window = (speaker_reward_moving_average + listener_reward_moving_average)/2


    if cfg.experiment.print_findings:
        print(f"Episodes: {episode:6},      Av R:{average_reward_moving_window:.2f}   Av S R: {speaker_reward_moving_average:.2f}, Av L R: {listener_reward_moving_average:.2f},  Av intent met: {speakers_intent_met_moving_average:.2f}, Av Suc MissUnd: {succ_misunderstandings_moving_average:.2f}")

    experiment_stats_dict:dict = dict()

    experiment_stats_dict = {}
    if cfg.experiment.agent_pairing == "two_teams" and len(agents) > 3:

        agent_teams = [[],[]]

        for i in range(len(agents)):
            agent_team_index = i % 2
            agent_teams[agent_team_index].append(agents[i])

        population_word_interpretation_alignment_dict_team_A = calculate_population_vocabulary_interpretation(
            agents=agent_teams[0], print_messages=cfg.experiment.print_findings)

        population_word_interpretation_alignment_dict_team_B = calculate_population_vocabulary_interpretation(
            agents=agent_teams[1], print_messages=cfg.experiment.print_findings)

        for key in population_word_interpretation_alignment_dict_team_A:
            experiment_stats_dict[key + " (Team A)"] = population_word_interpretation_alignment_dict_team_A[key]

        for key in population_word_interpretation_alignment_dict_team_B:
            experiment_stats_dict[key + " (Team B)"] = population_word_interpretation_alignment_dict_team_B[key]


    population_word_interpretation_alignment_dict = calculate_population_vocabulary_interpretation(agents=agents, print_messages=cfg.experiment.print_findings)
    experiment_stats_dict.update(population_word_interpretation_alignment_dict)

    experiment_stats_dict["Average Reward"] = float(average_reward_moving_window)

    experiment_stats_dict["Speaker's Intent Met Ratio"] = float(speakers_intent_met_moving_average)

    experiment_stats_dict["Successful Misunderstanding Ratio"] = succ_misunderstandings_moving_average


    return experiment_stats_dict



def get_reward_table(cfg:DictConfig) -> list:

    if cfg.experiment.reward_table == "random_simple":
        symmetric_flag = random.choice([True, False])
        if symmetric_flag:
            #     simple_game
            stag_hunt_rewards: list = [[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]]
        else:
            #     simple_game_different
            stag_hunt_rewards: list = [[[-1, -1], [1, 1]], [[1, 1], [-1, -1]]]

    elif cfg.experiment.reward_table == "random_simple_non_symmetric":
        symmetric_flag = random.choice([True, False])
        if symmetric_flag:
            #     simple_game_non_symmetric
            stag_hunt_rewards: list = [[[3, 3], [-3, -3]], [[-1, -1], [1, 1]]]
        else:
            #     simple_game_non_symmetric_different
            stag_hunt_rewards: list = [[[-3, -3], [3, 3]], [[1, 1], [-1, -1]]]

    elif cfg.experiment.reward_table == "random_3x3":
        random_pointer = random.randint(0, 2)
        if random_pointer == 0:
            stag_hunt_rewards: list = [[[1, 1], [-1, -1], [-1, -1]], [[-1, -1], [1, 1], [-1, -1]], [[-1, -1], [-1, -1], [1, 1]]]
        elif random_pointer == 1:
            stag_hunt_rewards: list = [[[-1, -1], [1, 1], [-1, -1]], [[-1, -1], [-1, -1], [1, 1]], [[1, 1], [-1, -1], [-1, -1]]]
        else:
            stag_hunt_rewards: list = [[[-1, -1], [-1, -1], [1, 1]], [[1, 1], [-1, -1], [-1, -1]], [[-1, -1], [1, 1], [-1, -1]]]

    else:
        raise ValueError(f"Unrecognized reward_table: {cfg.experiment.reward_table}")

    return stag_hunt_rewards
