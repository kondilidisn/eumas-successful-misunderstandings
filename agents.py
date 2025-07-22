import numpy as np

from utils import *
import string

class Agent:
    def __init__(self, cfg:DictConfig, interpretation_dimensions_per_role: list[int], population_size:Optional[int]= None):

        self.interpretation_dimensions_per_role: list[int] = interpretation_dimensions_per_role
        self.word_length = cfg.experiment.word_length
        self.softmax_word_role_interpretations = cfg.agent.softmax_word_role_interpretations

        self.interpretation_space_dimensionality = sum(self.interpretation_dimensions_per_role)

        self.interpret_dims_indices_per_role = \
            [sum(self.interpretation_dimensions_per_role[:i]) for i in
             range(len(self.interpretation_dimensions_per_role) + 1)]

        self.words: list[str] = []
        self.max_vocab_size = cfg.agent.max_vocab_size
        self.word_interpretations: Optional[npt.NDArray] = None
        self.word_last_used_counter: Optional[npt.NDArray] = None
        self.forget_word_after_interactions = cfg.agent.forget_word_threshold
        self.forget_oldest_word_or_least_useful: str = cfg.agent.forget_oldest_word_or_least_useful

        self.word_used_counter: Optional[npt.NDArray] = None

    #     parameters for epsilon greedy decay
        self.use_epsilon_greedy: bool = cfg.agent.use_epsilon_greedy
        self.epsilon_start: float = cfg.agent.epsilon_start
        self.epsilon_min: float = cfg.agent.epsilon_min

        if population_size is None:
            population_size = cfg.experiment.population_size

        self.epsilon_decay_steps: int = (cfg.experiment.episodes / (population_size - 1)) * cfg.agent.apply_epsilon_in_episode_ratio
        self.epsilon_decay_linear_step = (self.epsilon_start - self.epsilon_min) / self.epsilon_decay_steps
        self.current_epsilon: float = cfg.agent.epsilon_start
        self.episodes_played: int = 0

        self.speaker_actions_average_reward: npt.NDArray = np.zeros(self.interpretation_dimensions_per_role[0])
        self.speaker_actions_number_of_times_selected: npt.NDArray = np.zeros(self.interpretation_dimensions_per_role[0])

        self.average_reward_per_role:list[list[float]] = [[] for _ in range(len(self.interpretation_dimensions_per_role))]

        self.cfg:DictConfig = cfg

        self.learning_rate:float = cfg.agent.learning_rate


    def get_number_of_times_agent_played_per_role(self) -> tuple[int,int]:
        return (np.sum(self.speaker_actions_number_of_times_selected), np.sum(self.listeners_contextless_actions_number_of_times_selected))

    def add_reward_per_role_memory(self, role, reward):
        self.average_reward_per_role[role].append(reward)

    def get_mean_reward_per_role_and_reset(self):
        average_reward_per_role = [(sum(self.average_reward_per_role[i])/len(self.average_reward_per_role[i])) if len(self.average_reward_per_role[i]) > 0 else 0 for i in range(len(self.interpretation_dimensions_per_role))]
        self.average_reward_per_role = [[] for _ in range(len(self.interpretation_dimensions_per_role))]
        return average_reward_per_role

    def update_word_used(self, word_used:str):
        word_index:int = self.words.index(word_used)
        self.word_used_counter[word_index] += 1

    def update_epsilon(self):
        if self.episodes_played > self.epsilon_decay_steps:
            self.current_epsilon = self.epsilon_min
        else:
            self.current_epsilon -= self.epsilon_decay_linear_step
            if self.current_epsilon < 0:
                self.current_epsilon = self.epsilon_min
        self.episodes_played += 1

    def get_global_interpret_dim_given_role_and_local_dim(self, role: int, local_dim: int):
        return self.interpret_dims_indices_per_role[role] + local_dim

    def get_role_interpretation_space(self, role: int, word_index: Optional[int] = None) -> npt.NDArray:
        interpretation_space_of_role = self.word_interpretations[:,
                                       self.interpret_dims_indices_per_role[role]:self.interpret_dims_indices_per_role[
                                           role + 1]]
        if word_index is not None:
            return interpretation_space_of_role[word_index]
        else:
            return interpretation_space_of_role


    def update_word_interpretation_for_role_group(self, role: int, observed_dim: int, word: str, reward: float) -> None:
        used_word_index: int = self.words.index(word)

        global_observed_dim_index = self.get_global_interpret_dim_given_role_and_local_dim(role, observed_dim)

        self.word_interpretations[used_word_index][global_observed_dim_index] += reward * self.learning_rate



    def reset_words_used_counter_and_forget_words(self, words_used:list[str]):
        # increase last time each word was used counter and check if needed to forget a word
        # we go from higher index to lowest to be able to delete words at the same time
        for word_index in reversed(range(len(self.words))):
            if self.words[word_index] in words_used:
                self.word_last_used_counter[word_index] = 0
            else:
                self.word_last_used_counter[word_index] += 1
                if self.word_last_used_counter[word_index] > self.forget_word_after_interactions:
                    self.forget_word(word_index)

    def forget_word(self, word_index: int) -> None:
        del self.words[word_index]
        self.word_interpretations = np.delete(self.word_interpretations, (word_index), axis=0)
        self.word_last_used_counter = np.delete(self.word_last_used_counter, (word_index), axis=0)
        self.word_used_counter = np.delete(self.word_used_counter, (word_index), axis=0)

    def add_word(self, word: Optional[str] = None):

        # if the agent already uses full word-memory capacity, it needs to forget the one with the total returned reward
        if len(self.words) == self.max_vocab_size:

            if self.forget_oldest_word_or_least_useful == "least_useful":
                total_reward_per_word = np.sum(self.word_interpretations, axis=1)

                index_of_word_to_forget: int = int(np.argmax(total_reward_per_word))
            elif self.forget_oldest_word_or_least_useful == "oldest":
                index_of_word_to_forget: int = int(np.argmax(self.word_last_used_counter))

            else:
                raise ValueError("self.forget_oldest_word_or_least_useful unknown!: " + self.forget_oldest_word_or_least_useful)
            self.forget_word(index_of_word_to_forget)

        if word is None:
            new_word_form: str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=self.word_length))

            while new_word_form in self.words:
                new_word_form: str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=self.word_length))
        else:
            new_word_form: str = word

        self.words.append(new_word_form)

        if self.word_interpretations is None:
            self.word_interpretations = np.zeros((1, self.interpretation_space_dimensionality))
            self.word_last_used_counter = np.zeros((1, 1))
            self.word_used_counter = np.zeros((1, 1))
        else:
            self.word_interpretations = np.vstack(
                (self.word_interpretations, np.zeros((1, self.interpretation_space_dimensionality))))
            self.word_last_used_counter = np.vstack((self.word_last_used_counter, np.zeros((1, 1))))
            self.word_used_counter = np.vstack((self.word_used_counter, np.zeros((1, 1))))


    def select_word_to_communicate_give_role_and_observation(self, role: int, observed_dim: int, dont_update_epsilon:bool=False) -> str:

        word_index_to_communicate: int

        if len(self.words) == 0:
            self.add_word()
            word_index_to_communicate = len(self.words) - 1
        else:

            original_word_role_interpretations: npt.NDArray = self.get_role_interpretation_space(role)

            word_role_interpretations = np.copy(original_word_role_interpretations)

            # let's see which observation each word would describe, while breaking ties randomly
            # Find the maximum value in each row
            max_vals = np.max(word_role_interpretations, axis=1, keepdims=True)

            # Get all indices where each row has the maximum value
            max_indices = np.where(word_role_interpretations == max_vals)

            # Create an array to store the chosen indices
            word_dims_best_describe_dims_random_tie_braking = np.zeros(word_role_interpretations.shape[0], dtype=int)

            # Iterate over each row and randomly select one of the max indices
            for row in range(word_role_interpretations.shape[0]):
                row_indices = max_indices[1][max_indices[0] == row]  # Get indices for this row
                word_dims_best_describe_dims_random_tie_braking[row] = np.random.choice(row_indices)  # Randomly select one

            if observed_dim not in word_dims_best_describe_dims_random_tie_braking:
                self.add_word()
                word_index_to_communicate = len(self.words) - 1

            else:
                word_indices: list[int] = []
                word_scores: list[float] = []

                for word_index, best_described_dim in enumerate(word_dims_best_describe_dims_random_tie_braking):
                    if best_described_dim == observed_dim:
                        word_indices.append(word_index)
                        word_scores.append(float(original_word_role_interpretations[word_index][observed_dim]))

                if len(word_indices) == 1:
                    word_index_to_communicate = word_indices[0]
                else:

                    if self.use_epsilon_greedy:
                        # we apply epsilon greedy to select which word to communicate.
                        if not dont_update_epsilon:
                            self.update_epsilon()

                        softmaxed_scores = apply_softmax_per_row(np.array(word_scores))

                        epsilon_softmaxed_scores = (1 - self.current_epsilon) * softmaxed_scores + \
                                                (self.current_epsilon / len(word_indices)) * np.ones_like(softmaxed_scores)

                        word_index_to_communicate = int(np.random.choice(np.array(word_indices), 1, p=epsilon_softmaxed_scores.tolist()))

                    else:
                        word_scores_numpy = np.array(word_scores)

                        sub_index_of_word_to_communicate = np.random.choice(
                            np.flatnonzero(word_scores_numpy == word_scores_numpy.max()))

                        word_index_to_communicate:int = word_indices[int(sub_index_of_word_to_communicate)]


        if self.words[word_index_to_communicate] not in self.words:
            raise ValueError(f'{self.words[word_index_to_communicate]} not in {self.words}')

        return self.words[word_index_to_communicate]


    def get_average_score_of_word(self, word:str):
        return self.get_average_score_of_word_index(self.words.index(word))


    def get_average_score_of_word_index(self, word_index:int):
        return float(np.sum(self.word_interpretations[word_index], axis=0) / self.word_used_counter[word_index])


    def get_average_score_of_word_per_role(self, word:str, role:int):
        return self.get_average_score_of_word_index_per_role(self.words.index(word), role)

    def get_average_score_of_word_index_per_role(self, word_index:int, role:int):
        return float(np.sum(self.get_role_interpretation_space(role=role, word_index=word_index)) / self.word_used_counter[word_index])


    def update_speakers_action_expected_reward(self, selected_action:int, reward:float):

        self.speaker_actions_number_of_times_selected[selected_action] += 1
        self.speaker_actions_average_reward[selected_action] += reward * self.learning_rate

    def select_dimension_given_role_and_word(self, role: int, word: str, dont_update_epsilon:bool=False) -> int:

        selected_dimension: int

        if word not in self.words:
            self.add_word(word=word)
            role_dimensions = self.interpretation_dimensions_per_role[role]
            selected_dimension = random.randint(0, role_dimensions - 1)

        else:
            word_index: int = self.words.index(word)
            word_role_interpretations: npt.NDArray = self.get_role_interpretation_space(role, word_index)

            if self.softmax_word_role_interpretations:
                word_role_interpretations = apply_softmax_per_row(word_role_interpretations)

            if self.use_epsilon_greedy:

                if not dont_update_epsilon:
                    self.update_epsilon()

                word_role_interpretations = (1-self.current_epsilon)*word_role_interpretations + \
                                            (self.current_epsilon / self.interpretation_dimensions_per_role[role]) * np.ones_like(word_role_interpretations)

                selected_dimension = np.random.choice(np.arange(self.interpretation_dimensions_per_role[1]), p=word_role_interpretations)

            else:

                # break word interpretation ties randomly
                max_indices = np.where(word_role_interpretations == np.max(word_role_interpretations))[0]

                selected_dimension = np.random.choice(max_indices)

        return selected_dimension

    def print_vocabulary(self) -> None:
        for i in range(len(self.words)):
            print(self.words[i], self.word_interpretations[i])