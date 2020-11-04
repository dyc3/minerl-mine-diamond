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

checkpoint_dir = Path("ckpt/")
if not checkpoint_dir.exists():
    checkpoint_dir.mkdir()

from tensorflow import keras
from models import make_diamond_miner_model
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 2.5e-4

model = make_diamond_miner_model((64, 64, 3), (64,))
model_target = make_diamond_miner_model((64, 64, 3), (64,))
loss_function = keras.losses.Huber()
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss='mse', optimizer=optimizer)
model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

epsilon_start = 0.99
epsilon_min = 0.01
epsilon = epsilon_start
max_timesteps = 100000
explore_ts = max_timesteps * 0.8
gamma = 0.9
memory = []
memory_size = 300
batch_size = 32
train_interval = 4 # train the model every x frames
target_update_interval = 10000 # update the target model every x frames

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

    # pre train
    if (checkpoint_dir / "pretrain.h5").exists():
        print("Loading pretrain weights")
        model.load_weights(checkpoint_dir / "pretrain.h5")
    else:
        with alive_bar(title="pretrain", calibrate=120) as bar:
            for current_state, action, reward, next_state, done in data.batch_iter(batch_size=2, num_epochs=5, seq_len=32):
                loss = model.train_on_batch([current_state["pov"].reshape(-1, 64, 64, 3), current_state["vector"].reshape(-1, 64)], action["vector"].reshape(-1, 64))
                bar.text(f"loss: {loss}")
                bar()
        model.save_weights(checkpoint_dir / "pretrain.h5")
    model_target.set_weights(model.get_weights())

    env.make_interactive(port=6666)

    aicrowd_helper.training_start()
    frame_count = 0
    episodes = 1024
    for episode in range(episodes):
        if (checkpoint_dir / f"episode-{episode}.h5").exists():
            if not (checkpoint_dir / f"episode-{episode + 1}.h5").exists():
                model.load_weights(checkpoint_dir / f"episode-{episode}.h5")
            if epsilon > epsilon_min:
                epsilon -= (epsilon_start - epsilon_min) / explore_ts
            frame_count += 6000
            continue

        obs = env.reset()
        done = False
        netr = 0

        epoch_loss = []
        with alive_bar(title=f"episode: {episode}") as bar:
            while not done:
                explore = np.random.rand() < epsilon
                if explore:
                    bar.text("perform action: explore")
                    action = env.action_space.sample()
                else:
                    bar.text("perform action: predict")
                    action = env.action_space.noop()
                    action["vector"] = model.predict([obs["pov"].reshape(-1, 64, 64, 3), obs["vector"].reshape(-1, 64)])[0]
                new_obs, reward, done, info = env.step(action)
                netr += reward

                memory.append((obs, action, reward, new_obs, done))
                # Make sure we restrict memory size to specified limit
                if len(memory) > memory_size:
                    memory.pop(0)

                if frame_count % train_interval == 0:
                    bar.text("training: build replay")
                    replay = random.sample(memory, min(batch_size, len(memory)))
                    states_pov = np.array([a[0]["pov"] for a in replay]).reshape(-1, 64, 64, 3)
                    states_vector = np.array([a[0]["vector"] for a in replay]).reshape(-1, 64)
                    # new_states_pov = np.array([a[3]["pov"] for a in replay]).reshape(-1, 64, 64, 3)
                    # new_states_vector = np.array([a[3]["vector"] for a in replay]).reshape(-1, 64)

                    # Predict the expected utility of current state and new state
                    bar.text("training: predict Q")
                    Q = model_target.predict([states_pov, states_vector])
                    Q_new = [a[2] for a in replay] + gamma * tf.reduce_max(
                        Q, axis=1
                    )

                    # masks = tf.one_hot([a[1]["vector"] for a in replay], 64)

                    bar.text("training: backprop")
                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = model([states_pov, states_vector])

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        # q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        q_action = tf.reduce_sum(q_values, axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(Q_new, q_action)
                        grads = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    epoch_loss.append(loss)

                if epsilon > epsilon_min:
                    epsilon -= (epsilon_start - epsilon_min) / explore_ts
                print("explore:", explore, "net reward:", netr, "loss:", loss, "epsilon:", epsilon)
                bar()
                obs = new_obs
                if frame_count % target_update_interval == 0:
                    print("updated target model")
                    model_target.set_weights(model.get_weights())
                frame_count += 1
        model.save_weights(checkpoint_dir / f"episode-{episode}.h5")

        aicrowd_helper.register_progress(episode / episodes)

    # Save trained model to train/ directory
    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    aicrowd_helper.training_end()
    env.close()


if __name__ == "__main__":
    main()
