"""
train_qlearning.py
--------------------
Trains a QLearningScheduler across many practice episodes. Each episode
is a fresh, independently-random world (different emitter timing) - so
the learned Q-table reflects a GENERAL pattern ("band X tends to be hot
at phase Y of the cycle") rather than memorizing one specific episode.

This is kept separate from simulate.py's run_simulation() because
training needs to call scheduler.update() after every scan, which a
normal (non-learning, or already-trained) scheduler never needs.
"""

import random
from environment import RFEnvironment
from receiver import Receiver
from schedulers import QLearningScheduler
from simulate import make_default_emitters


def train_qlearning(num_bands: int, num_steps: int, num_episodes: int,
                     p_detect: float = 0.9, p_false_alarm: float = 0.05,
                     period: int = None, training_epsilon: float = 0.15,
                     alpha: float = 0.1, gamma: float = 0.5,
                     seed_offset: int = 0) -> QLearningScheduler:
    """
    Runs num_episodes practice simulations, updating one shared Q-table
    throughout. Returns a QLearningScheduler with epsilon=0 (ready for
    evaluation/deployment) and its learned q_table already filled in.

    seed_offset lets us pick which block of random seeds to train on,
    so we can later evaluate on a DIFFERENT block of seeds the agent
    has never seen - the standard "held-out test set" idea, borrowed
    from ML practice, applied here to confirm the agent generalizes
    rather than memorizing.
    """
    scheduler = QLearningScheduler(num_bands, period=period, epsilon=training_epsilon)

    for episode in range(num_episodes):
        seed = seed_offset + episode
        random.seed(seed)
        emitters = make_default_emitters(num_bands)
        env = RFEnvironment(num_bands, emitters)
        receiver = Receiver(num_bands, p_detect=p_detect, p_false_alarm=p_false_alarm, seed=seed + 5000)

        for t in range(num_steps):
            truth_row = env.step(t)
            state = t % scheduler.period
            band = scheduler.choose_band(t, receiver)   # epsilon-greedy during training
            hit = receiver.scan(t, band, truth_row)
            reward = 1.0 if hit else 0.0
            next_state = (t + 1) % scheduler.period
            scheduler.update(state, band, reward, next_state, alpha=alpha, gamma=gamma)

    # Freeze the policy for deployment: stop exploring, always exploit
    # the learned Q-table from here on.
    scheduler.epsilon = 0.0
    return scheduler