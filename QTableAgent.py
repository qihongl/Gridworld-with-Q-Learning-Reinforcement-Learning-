import numpy as np


class QTableAgent():
    # Intialise
    def __init__(
            self,
            environment,
            alpha=0.1, gamma=1
    ):
        self.environment = environment
        # Store all Q-values in dictionary of dictionaries
        self.q_table = dict()
        # Loop through all possible grid spaces, create sub-dictionary for each
        for x in range(environment.height):
            for y in range(environment.width):
                # Populate sub-dictionary with zero values for possible moves
                self.q_table[(x, y)] = {
                    'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0
                }
        self.alpha = alpha
        self.gamma = gamma

    def pick_action(self, available_actions, epsilon=.05):
        """Returns the optimal action from Q-Value table.
        If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon.
        """
        if np.random.uniform(0, 1) < epsilon:
            action = available_actions[np.random.randint(
                0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[self.environment.current_location]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice(
                [k for k, v in q_values_of_state.items() if v == maxValue]
            )

        return action

    def learn(self, old_state, reward, new_state, action):
        """Updates the Q-value table using Q-learning"""
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]
        # Q(S,A) <-  (1-alpha) Q(S,A) + alpha (R + gamma max Q(S'))
        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value +\
            self.alpha * (reward + self.gamma * max_q_value_in_new_state)
