# Simple env test.
# HACK: fixes strange "Failed to get convolution algorithm." error. If something breaks, try removing this configuration
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import json
import select
import time
import logging
import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from alive_progress import alive_bar

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser

import coloredlogs
coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

epsilon_start = 0.99
epsilon_min = 0.01
epsilon = epsilon_start
max_timesteps = 10000
explore_ts = max_timesteps * 0.8
gamma = 0.9
memory = []
memory_size = 300
batch_size = 32

def main():
    global epsilon
    global memory
    """
    This function will be called for training phase.
    """
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)

    # Sample code for illustration, add your training code below
    env = gym.make(MINERL_GYM_ENV)

    env.make_interactive(port=6666, realtime=True)

    aicrowd_helper.training_start()
    episodes = 1024
    trajectory = data.load_data("v3_excellent_pluot_behemoth-4_3461-4804")
    for episode in range(episodes):
        obs = env.reset()
        done = False
        netr = 0

        with alive_bar(title=f"episode: {episode}") as bar:
            bar.text("replaying trajectory")
            for state, action, reward, next_state, done in trajectory:
                obs, reward, done, info = env.step(action)
                bar()
            i = 0
            bar.text("testing inputs")
            while not done:
                print(i % 64)
                action = env.action_space.noop()
                vec = np.zeros((64,))
                vec[i % 64] = -0.5
                action["vector"] = vec
                obs, reward, done, info = env.step(action)
                netr += reward
                bar()
                i += 1

        aicrowd_helper.register_progress(episode / episodes)

    # Save trained model to train/ directory
    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    aicrowd_helper.training_end()
    env.close()


if __name__ == "__main__":
    main()
