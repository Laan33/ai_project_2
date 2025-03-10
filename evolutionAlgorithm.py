import play_game
import random
"""
Evolutionary Algorithm for the Prisoner's Dilemma.
• Population size: 50-100 individuals
• Tournament selection
• Maintain diversity in the population
• Explore convergence with different fitness evaluations (different portions
of all-c, all-d, other...)
"""

population_size = 75

choices = ['C', 'D'] # Cooperate or Defect
# i) Rather than a choosing a binary alphabet, we could have real values indicating the probabilities of cooperating. Mutation and crossover would have to be modified.
# ii) Rather than considering a memory of length one, you could increase the memory length - for example, is a memory of 2 was used, we would have a genotype of length 5 ( and hence a strategy space of 2^5). This would be what to do on the first move, what to do following CC, CD, DC, DD by the opponent. This can be generalised. One could also include one's own moves too in the memory condition.


# # Create initial random population
# def initialize_population(pop_size):
#     return [Strategy() for _ in range(pop_size)]

import numpy as np

class IPDGenotype:
    def __init__(self, memory_depth=3):
        self.memory_depth = memory_depth
        self.strategy_length = 2 ** memory_depth  # All possible opponent history patterns
        self.strategy = np.random.rand(self.strategy_length)  # Probabilities of cooperation (0 to 1)

    def play(self, opponent_history):
        """Decide whether to cooperate (C) or defect (D) based on opponent's past moves."""
        if len(opponent_history) < self.memory_depth:
            history_index = 0  # Default to first entry if not enough history
        else:
            history_index = int("".join('0' if move == 'C' else '1' for move in opponent_history[-self.memory_depth:]), 2)

        # Chance of

        return 'C' if np.random.rand() < self.strategy[history_index] else 'D'  # Probabilistic decision

agent = IPDGenotype(memory_depth=3)
opponent = play_game.adaptive_strategy

# Define a wrapper class for the agent's play function
class StrategyWrapper:
    def __init__(self, strategy_function):
        self.strategy_function = strategy_function

    def play(self, history):
        return self.strategy_function(history)

# Create instances of the wrapper class for both strategies
agent_strategy = StrategyWrapper(agent.play)
opponent_strategy = StrategyWrapper(opponent)

for _ in range(5):
    # Play the game using the wrapped strategies
    agentScore, opponentScore, history1, history2 = play_game.play_game(agent_strategy, opponent_strategy, num_rounds=15)
    print(f"Agent score: {agentScore}")
    print(f"Opponent score: {opponentScore}")
    print(f"Agent history: {history1}")

