import pickle

import matplotlib.pyplot as plt
import numpy as np

for i in range(4):
    statistics_file = f'statistics_{i}.pkl'

    # open the pickle file
    with open(statistics_file, 'rb') as f:
        statistics = pickle.load(f)

    # plot the reward
    y = np.array(statistics['mean_episode_rewards'])
    # plot dotted line with color based on index i
    plt.plot(y / y[~np.isnan(y)].max(), label=f'mean-{i}', linestyle='--', color=f'C{i}')
    z = np.array(statistics['best_mean_episode_rewards'])
    plt.plot(z / z.max(), label=f'best-{i}', linestyle='-', color=f'C{i}')

plt.title('Mean Episode Reward (Normalized) by Time Step on Different Games')
plt.xlabel('Steps')
plt.ylabel('Reward')

plt.legend()

plt.show()
# plt.savefig('reward.png')
plt.close()
