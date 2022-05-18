import os
import sys

import gym
import torch.optim as optim

from dqn_learn import OptimizerSpec, dqn_learing
from dqn_model import DQN
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


def main(config_number, schedule_timesteps=2000000, final_p=0.0, task_num=0):
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[task_num]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed, config_number)

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= task.max_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(schedule_timesteps, final_p)

    dqn_learing(
        config_number,
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ,
    )


if __name__ == '__main__':
    CONDA_ENV_PATH = ""
    os.environ['PATH'] = os.environ['PATH'] + ':' + CONDA_ENV_PATH

    if len(sys.argv) > 1:
        config_number = int(sys.argv[1])
    else:
        config_number = 0

    # run_configs = [
    #     {
    #         'final_p': 0.0,
    #     },
    #     {
    #         'final_p': 0.05,
    #     },
    #     {
    #         'final_p': 0.1,
    #     },
    #     {
    #         'final_p': 0.2,
    #     },
    # ]

    run_configs = [
        {
            'task_num': 3,
        },
        {
            'task_num': 0,
        },
        {
            'task_num': 1,
        },
        {
            'task_num': 2,
        },
        {
            'task_num': 6,
        },
    ]

    main(config_number, **run_configs[config_number])
