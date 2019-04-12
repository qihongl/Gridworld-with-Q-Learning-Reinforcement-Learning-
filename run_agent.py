from GridWorld import GridWorld
from QTableAgent import QTableAgent as Agent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='talk', palette='colorblind')


def play(
        env, agent, epsilon,
        trials=500, max_steps=1000, learn=False
):
    """
    train the agent
    """
    # Initialise performance log
    log_return = []
    log_steps = []

    for trial in range(trials):
        # Initialise values of each game
        cumulative_reward = 0
        step = 0
        game_over = False
        # run the agent
        while step < max_steps and not game_over:
            old_state = env.current_location
            action = agent.pick_action(env.actions, epsilon)
            reward = env.make_step(action)
            new_state = env.current_location

            if learn:  # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state, action)

            # update R and n steps
            cumulative_reward += reward
            step += 1

            if env.check_state() == 'TERMINAL':
                env.__init__()
                game_over = True

        # Append reward for current trial to performance log
        log_return.append(cumulative_reward)
        log_steps.append(step)
    return log_return, log_steps


# define the env and agent
epsilon = 0.05
alpha = 0.1
gamma = .9
grid_world = GridWorld()
agent = Agent(
    grid_world,
    alpha=alpha, gamma=gamma
)

# run the Q-Agent
n_trials = 150
log_return, log_steps = play(
    grid_world, agent, epsilon,
    trials=n_trials, learn=True
)

'''
learning curve
'''

f, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

axes[0].plot(log_return)
axes[0].axhline(0, color='grey', linestyle='--')
axes[0].set_title('Learning curve')
axes[0].set_ylabel('Return')

axes[1].plot(log_steps)
axes[1].set_title(' ')
axes[1].set_ylabel('n steps took')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim([0, None])

sns.despine()
f.tight_layout()
