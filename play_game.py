
# Two prisoner's dilemma

# payoff matrix:
#    |   C    |   D  |
# --------------------
# C  | (3,3)  | (0,5)|
# --------------------
# D  | (5,0)  | (1,1)|
# --------------------

# C = cooperate
# D = defect

import random

import numpy as np

PAYOFF_MATRIX = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

# Standard strategies to play: Always cooperate, Always defect, Tit-for-Tat (do what opponent did last round), other
def always_cooperate(history):
    return 'C'

def always_defect(history):
    return 'D'

def tit_for_tat(history):
    if not history:
        return 'C'
    return history[-1]

def random_strategy(history):
    return random.choice(['C', 'D'])

#Implements a strategy that defects if the opponent has defected more than twice in the last 5 rounds, otherwise cooperates
def adaptive_strategy(history, threshold=2, window=5):
    if len(history) < window:
        recent_history = history
    else:
        recent_history = history[-window:]
    opponent_defects = sum(1 for move in recent_history if move == 'D')
    if opponent_defects > threshold:
        return 'D'
    else:
        return 'C'

def spiteful_strategy(history, defect_chance = 0.05, first_move_forgive = True):
    if len(history) < 1:
        return 'C' if np.random.rand() > 0.3 else 'D'  # 0.3 probability that it defects
    if first_move_forgive is False:
        opponent_defects = sum(1 for move in history if move == 'D')
    else:
        opponent_defects = sum(1 for move in history[1:] if move == 'D')
    return 'C' if np.random.rand() > (defect_chance * opponent_defects) else 'D'  # Probabilistic decision


def play_game(strategy1, strategy2, num_rounds=90):
    history1 = []
    history2 = []
    score1 = 0
    score2 = 0

    for _ in range(num_rounds):
        move1, move2, payoff1, payoff2 = play_round(strategy1, strategy2, history1, history2)

        history1.append(move2)
        history2.append(move1)

        score1 += payoff1
        score2 += payoff2

    return score1, score2, history1, history2

# Function to play a single round of the Prisoner's Dilemma
def play_round(strategy1, strategy2, history1, history2):
    move1 = strategy1.play(history1)
    move2 = strategy2.play(history2)

    payoff1, payoff2 = PAYOFF_MATRIX[(move1, move2)]
    return move1, move2, payoff1, payoff2


