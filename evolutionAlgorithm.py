from matplotlib import pyplot as plt
import numpy as np
import itertools
from math import isclose

import play_game as game
import random

from play_game import play_game

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



class IPDGenotype:
    def __init__(self, memory_depth=2):
        self.memory_depth = memory_depth
        self.strategy_length = (memory_depth ** 2) + 3  # (2^2) + 1 + 2 for memory=2
        self.strategy = np.random.rand(self.strategy_length)  # Probabilities of cooperating

        # Generate all history combinations
        self.history_map = self.create_history_map()

    def create_history_map(self):
        """Creates a mapping of history states to indices in the strategy array."""
        history_map = {}

        # No history case
        history_map[""] = 0

        # Single move history cases
        history_map["C"] = 1
        history_map["D"] = 2

        # Full history cases (e.g., CC, CD, DC, DD for memory=2)
        all_combos = itertools.product("CD", repeat=self.memory_depth)  # Generate all history pairs
        index = 3  # Start after no-history and single-history cases

        for combo in all_combos:
            history_map["".join(combo)] = index
            index += 1

        return history_map

    def encode_history(self, opponent_history):
        """Encodes opponent's history into a strategy index."""
        history_str = "".join(move for move in opponent_history[-self.memory_depth:])
        return self.history_map.get(history_str, 0)  # Default to 'no history' if not found

    def play(self, opponent_history):
        """Decide whether to cooperate (C) or defect (D) based on opponent's past moves."""
        history_index = self.encode_history(opponent_history[-self.memory_depth:])
        return 'C' if np.random.rand() < self.strategy[history_index] else 'D'  # Probabilistic decision


class EvolutionaryIPD:
    def __init__(self, population_size=50, memory_depth=2, mutation_rate=0.1, opponentStrategy = game.always_defect):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [IPDGenotype(memory_depth) for _ in range(population_size)]
        self.opponentStrategy = opponentStrategy
        # self.fixed_strategies = [game.always_cooperate, game.always_defect, game.tit_for_tat, game.adaptive_strategy]

    def evaluate_fitness(self, agent):
        """Plays against fixed strategies and returns total score."""
        total_score = 0
        for _ in range(3):
            agent_score, _, _, _ = game.play_game(agent, StrategyWrapper(self.opponentStrategy), num_rounds=50)
            total_score += agent_score
        return total_score

    def tournament_selection(self, k=6):
        """Selects the best agent from a random subset of the population."""
        tournament = random.sample(self.population, k)
        return max(tournament, key=lambda agent: self.evaluate_fitness(agent))

    def crossover(self, parent1, parent2):
        """Creates a child by combining parent strategies (uniform crossover)."""
        child = IPDGenotype(parent1.memory_depth)
        for i in range(len(parent1.strategy)):
            child.strategy[i] = random.choice([parent1.strategy[i], parent2.strategy[i]])
        return child

    def mutate(self, agent):
        """Mutates agent strategy slightly."""
        for i in range(len(agent.strategy)):
            if np.random.rand() < self.mutation_rate:
                agent.strategy[i] += np.random.normal(0, 0.1)
                agent.strategy[i] = np.clip(agent.strategy[i], 0, 1)

    def evolve(self, generations=50):
        fitness_scores = []
        diversity_scores = []

        for gen in range(generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population
            best_fitness = max([self.evaluate_fitness(agent) for agent in self.population])
            fitness_scores.append(best_fitness)
            diversity_scores.append(calculate_diversity(self.population))  # Track diversity

            # Early stopping if no improvement
            if len(fitness_scores) > 50:
                if isclose(best_fitness, fitness_scores[-49], rel_tol=0.0025):
                    print(f'Stopping early at generation {gen} due to no improvement in fitness.')
                    break

        return max(self.population, key=lambda agent: self.evaluate_fitness(agent)), fitness_scores, diversity_scores

def calculate_diversity(population):
    """Measure diversity using standard deviation of strategy probabilities."""
    strategies = np.array([agent.strategy for agent in population])
    return np.mean(np.std(strategies, axis=0))  # Average standard deviation across genes

# Plot fitness progress over generations
def plot_fitness_over_time(fitness_scores, opponentStrategy):
    plt.plot(fitness_scores)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Time, Opponent: " + str(opponentStrategy.__name__))
    plt.show()

def plot_diversity(fitness_scores, diversity_scores, opponentStrategy):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color='tab:blue')
    ax1.plot(fitness_scores, label="Fitness", color='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Diversity", color='tab:red')
    ax2.plot(diversity_scores, label="Diversity", color='tab:red')

    plt.title("Fitness and Diversity Over Time (Opponent: " + str(opponentStrategy.__name__) + ")")
    fig.tight_layout()
    plt.show()


# Define a wrapper class for the agent's play function
class StrategyWrapper:
    def __init__(self, strategy_function):
        self.strategy_function = strategy_function

    def play(self, history):
        return self.strategy_function(history)


fixed_strategies = [game.always_cooperate, game.always_defect, game.tit_for_tat, game.adaptive_strategy]
best_agents = []


# # Example usage
# agent = IPDGenotype(memory_depth=2)
# print("History Mapping:", agent.history_map)  # Check if history encoding is correct
#
# opponent_history = ["C", "D"]  # Example history
# decision = agent.play(opponent_history)
# print(f"Agent decides to {'Cooperate' if decision == 'C' else 'Defect'}")

for strat in fixed_strategies:
    print("Fixed strategy: " + str(strat.__name__))
    evolution = EvolutionaryIPD(population_size=50, memory_depth=3, opponentStrategy=strat)
    best_agent, fitness_scores, diversity_scores = evolution.evolve(generations=300)
    best_agents.append([best_agent, strat.__name__])

    plot_diversity(fitness_scores, diversity_scores, strat)

    agentScore, fixedScore, _, _ = play_game(strategy1=best_agent, strategy2=StrategyWrapper(strat), num_rounds=50)
    print(f"Agent score: {agentScore}, Strat score: {fixedScore}")
    print(f"Best agent genome: {best_agent.strategy}")


print("\nTesting the agents on all the fixed strategies")
for best_agent in best_agents:
    print("--------------\n")
    for strat in fixed_strategies:
        print(f"Agent opponent: {best_agent[1]} Fixed strategy: {str(strat.__name__)}")
        agentScore, fixedScore, _, _ = play_game(strategy1=best_agent[0], strategy2=StrategyWrapper(strat), num_rounds=50)
        print(f"Agent score: {agentScore}, Strat score: {fixedScore}\n")


