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
    def __init__(self, memory_depth=2):
        self.memory_depth = memory_depth
        self.strategy_length = 2 ** (memory_depth * 2)  # Considering opponent's last moves (CC, CD, DC, DD)
        self.strategy = np.random.rand(self.strategy_length)  # Probabilities of cooperation (0 to 1)

    def play(self, opponent_history):
        """Decide whether to cooperate (C) or defect (D) based on opponent's past moves."""
        if len(opponent_history) < self.memory_depth:
            history_index = 0  # Default to first entry if not enough history
        else:
            history_index = int("".join('0' if move == 'C' else '1' for move in opponent_history[-self.memory_depth:]), 2)

        return 'C' if np.random.rand() < self.strategy[history_index] else 'D'  # Probabilistic decision


class EvolutionaryIPD:
    def __init__(self, population_size=50, memory_depth=2, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [IPDGenotype(memory_depth) for _ in range(population_size)]

    def evaluate_fitness(self, agent, opponent_pool):
        """Plays against multiple opponents and returns a fitness score."""
        total_score = 0
        for opponent in opponent_pool:
            agent_wrapper = StrategyWrapper(agent.play)
            opponent_wrapper = StrategyWrapper(opponent.play)
            agent_score, _, _, _ = play_game.play_game(agent_wrapper, opponent_wrapper, num_rounds=15)
            total_score += agent_score
        return total_score

    def tournament_selection(self, k=5):
        """Selects the best agent from a random subset of the population."""
        tournament = random.sample(self.population, k)
        return max(tournament, key=lambda agent: self.evaluate_fitness(agent, self.population))

    def crossover(self, parent1, parent2):
        """Performs uniform crossover between two parents to create an offspring."""
        child = IPDGenotype(parent1.memory_depth)
        for i in range(len(parent1.strategy)):
            child.strategy[i] = random.choice([parent1.strategy[i], parent2.strategy[i]])  # Uniform crossover
        return child

    def mutate(self, agent):
        """Mutates an agent's strategy slightly."""
        for i in range(len(agent.strategy)):
            if np.random.rand() < self.mutation_rate:
                agent.strategy[i] += np.random.normal(0, 0.1)  # Small Gaussian mutation
                agent.strategy[i] = np.clip(agent.strategy[i], 0, 1)  # Keep probability in [0,1]

    def evolve(self, generations=50):
        """Runs the evolutionary algorithm over multiple generations."""
        for gen in range(generations):
            print(f"Generation {gen + 1}")

            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(agent, self.population) for agent in self.population]

            # Select the next generation
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population  # Update population


# Define a wrapper class for the agent's play function
class StrategyWrapper:
    def __init__(self, strategy_function):
        self.strategy_function = strategy_function

    def play(self, history):
        return self.strategy_function(history)


evolution = EvolutionaryIPD(population_size=50, memory_depth=2)
evolution.evolve(generations=50)



