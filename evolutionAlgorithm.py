from matplotlib import pyplot as plt
import numpy as np
import itertools
from math import isclose

from numpy.ma.extras import average
from openpyxl.styles.builtins import total

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
    def __init__(self, memory_depth=1):
        self.memory_depth = memory_depth
        self.strategy_length = 3 ** memory_depth + 1  # Number of possible history states + 1 for no history
        self.strategy = np.random.rand(self.strategy_length)  # Probabilities of cooperating

        # Generate all history combinations
        self.history_map = self.create_history_map()
        # print(self.history_map)

    def create_history_map(self):
        """Creates a mapping of history states to indices in the strategy array."""
        history_map = {"": 0} # No history

        # All combinations of "C" and "D" of length memory_depth (e.g. "C", "D", "CC", "CD", "DC", "DD" for memory_depth=2)
        for i in range(self.memory_depth + 1):
            for combo in itertools.product(choices, repeat=i):
                history_map["".join(combo)] = len(history_map)

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
    def __init__(self, population_size=50, memory_depth=2, mutation_rate=0.1, opponent_strategy = None, generalised_eval = False):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generalised_eval = generalised_eval
        self.population = [IPDGenotype(memory_depth) for _ in range(population_size)]
        self.fixed_strategies = [game.always_cooperate, game.always_defect, game.tit_for_tat, game.adaptive_strategy]
        if opponent_strategy is None:
            self.setStrat = False
            self.opponentStrategy = random.choice(self.fixed_strategies)
        else:
            self.setStrat = True
            self.opponentStrategy = opponent_strategy

    def evaluate_fitness(self, agent):
        """Plays against fixed strategies and returns total score."""
        # total_score = 0

        if self.setStrat is False:
            self.opponentStrategy = random.choice(self.fixed_strategies)
            agent_score, _, _, _ = game.play_game(agent, StrategyWrapper(self.opponentStrategy), num_rounds=50)
            return agent_score
        else:
            # for _ in range(3): # Getting a more balance fitness evaluation
            agent_score, _, _, _ = game.play_game(agent, StrategyWrapper(self.opponentStrategy), num_rounds=50)
            return agent_score
                # total_score += agent_score
                # return round(total_score/3, 2)

    def evaluate_fitness_all_fixed(self, agent):
        total_score = 0
        for strat in self.fixed_strategies:
            num_rounds = 50
            agent_score, _, agent_history, _ = game.play_game(agent, StrategyWrapper(strat), num_rounds=num_rounds)
            total_score += (agent_score + self.good_samaritan_award(agent_history, num_rounds=num_rounds))
        return round(total_score/len(self.fixed_strategies), 2)

    def good_samaritan_award(self, agent_history, num_rounds=50):
        """Award for cooperating with all fixed strategies, awarded 55% of the time."""
        if np.random.rand() <= 0.55 and all(move == 'C' for move in agent_history):
            return num_rounds * 3.5
        return 0

    def tournament_selection(self, k=5):
        """Selects the best agent from a random subset of the population."""
        tournament = random.sample(self.population, k)
        if self.generalised_eval is True:
            return max(tournament, key=lambda agent: self.evaluate_fitness_all_fixed(agent))
        else:
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
            if self.generalised_eval is True:
                best_fitness = max([self.evaluate_fitness_all_fixed(agent) for agent in self.population])
            else:
                best_fitness = max([self.evaluate_fitness(agent) for agent in self.population])

            fitness_scores.append(best_fitness)
            diversity_scores.append(calculate_diversity(self.population))  # Track diversity

            # # Early stopping if no improvement
            # if len(fitness_scores) > 150:
            #     if isclose(best_fitness, fitness_scores[-59], rel_tol=0.00015):
            #         print(f'Stopping early at generation {gen} due to no improvement in fitness.')
            #         break

        if self.generalised_eval is True:
            return max(self.population, key=lambda agent: self.evaluate_fitness_all_fixed(agent)), fitness_scores, diversity_scores
        else:
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

def plot_diversity(fitness_scores, diversity_scores, oppStrat=None):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color='tab:blue')
    ax1.plot(fitness_scores, label="Fitness", color='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Diversity", color='tab:red')
    ax2.plot(diversity_scores, label="Diversity", color='tab:red')

    if oppStrat is None:
        plt.title("Fitness and Diversity Over Time")
    else:
        plt.title("Fitness and Diversity Over Time, Opponent: " + str(oppStrat.__name__))
    fig.tight_layout()
    plt.show()

def print_history_strat_map(agent):
    for history, index in agent.history_map.items():
        print(f"History: {history}, Strategy Value: {round(agent.strategy[index], 2)}")

# Define a wrapper class for the agent's play function
class StrategyWrapper:
    def __init__(self, strategy_function):
        self.strategy_function = strategy_function

    def play(self, history):
        return self.strategy_function(history)


# ------------------

fixed_strategies = [game.always_cooperate, game.always_defect, game.tit_for_tat, game.adaptive_strategy, game.spiteful_strategy]
best_agents = []

evolution = EvolutionaryIPD(population_size=50, memory_depth=6, generalised_eval=True)
best_agent, fitness_scores, diversity_scores = evolution.evolve(generations=300)

plot_diversity(fitness_scores, diversity_scores)

print(f"Average fitness: {round(average(fitness_scores),2)}\n")
print("Best generalised agent strategy genome:")
print_history_strat_map(best_agent)

print("\n----------------")
print("Testing the generalised agent on all the fixed strategies")
print("----------------\n")
for strat in fixed_strategies:
    agentScore, fixedScore, _, _ = play_game(strategy1=best_agent, strategy2=StrategyWrapper(strat), num_rounds=50)
    print(f"Generalised agent score: {agentScore}, Strat score: {fixedScore}, Strat: {strat.__name__}")



