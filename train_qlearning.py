"""
train_qlearning.py
------------------

Trains the observation-aware Q-learning scheduler.

Unlike the original phase-only implementation, the RL state now
contains information derived from the receiver's actual observations.

The scheduler never receives the RF truth grid.
"""

import random

from environment import RFEnvironment
from receiver import Receiver
from schedulers import QLearningScheduler
from simulate import make_default_emitters


def train_qlearning(
    num_bands: int,
    num_steps: int,
    num_episodes: int,

    p_detect: float = 0.9,
    p_false_alarm: float = 0.05,

    period: int = None,

    training_epsilon: float = 0.15,

    alpha: float = 0.1,
    gamma: float = 0.5,

    seed_offset: int = 0,

    max_hit_age: int = 8
) -> QLearningScheduler:

    """
    Train one shared Q-learning agent over multiple episodes.

    State:

        (time phase,
         most recently detected band,
         age of most recent detection)

    Action:

        choose one RF band

    Reward:

        +1 for a receiver-reported HIT
         0 for a receiver-reported MISS

    The training process never gives the scheduler the truth grid.
    """

    # -------------------------------------------------------------
    # Create one shared Q-learning agent.
    # -------------------------------------------------------------
    scheduler = QLearningScheduler(
        num_bands=num_bands,
        period=period,
        epsilon=training_epsilon,
        max_hit_age=max_hit_age
    )

    # -------------------------------------------------------------
    # Training episodes
    # -------------------------------------------------------------
    for episode in range(num_episodes):

        seed = seed_offset + episode

        # Make the episode reproducible.
        random.seed(seed)

        # Create a fresh RF environment.
        emitters = make_default_emitters(num_bands)

        env = RFEnvironment(
            num_bands,
            emitters
        )

        # Create a fresh receiver.
        receiver = Receiver(
            num_bands,
            p_detect=p_detect,
            p_false_alarm=p_false_alarm,
            seed=seed + 5000
        )

        # ---------------------------------------------------------
        # Run one episode
        # ---------------------------------------------------------
        for t in range(num_steps):

            # -----------------------------------------------------
            # STATE BEFORE ACTION
            #
            # This comes ONLY from receiver history.
            # -----------------------------------------------------
            state = scheduler.get_state(
                t,
                receiver
            )

            # -----------------------------------------------------
            # Environment advances.
            #
            # The scheduler itself does NOT see the truth grid.
            # -----------------------------------------------------
            truth_row = env.step(t)

            # -----------------------------------------------------
            # Choose ONE band.
            # -----------------------------------------------------
            band = scheduler.choose_band(
                t,
                receiver
            )

            # -----------------------------------------------------
            # Receiver scans the selected band.
            #
            # The receiver returns what it OBSERVED.
            # -----------------------------------------------------
            hit = receiver.scan(
                t,
                band,
                truth_row
            )

            # -----------------------------------------------------
            # Reward
            #
            # We deliberately use only the receiver's observation.
            #
            # HIT  -> +1
            # MISS ->  0
            # -----------------------------------------------------
            reward = 1.0 if hit else 0.0

            # -----------------------------------------------------
            # STATE AFTER ACTION
            #
            # The receiver now contains the newest observation.
            # Therefore the next state can be different.
            # -----------------------------------------------------
            next_state = scheduler.get_state(
                t + 1,
                receiver
            )

            # -----------------------------------------------------
            # Q-learning update
            # -----------------------------------------------------
            scheduler.update(
                state=state,
                action=band,
                reward=reward,
                next_state=next_state,
                alpha=alpha,
                gamma=gamma
            )

    # -------------------------------------------------------------
    # Training finished.
    #
    # Disable exploration for evaluation/deployment.
    # -------------------------------------------------------------
    scheduler.epsilon = 0.0

    return scheduler