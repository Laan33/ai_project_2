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

choices = ['C', 'D']



# Create initial random population
def initialize_population(pop_size):
    return [Strategy() for _ in range(pop_size)]

#

class Strategy:
    def __init__(self, strategy_function=None):
        if strategy_function is None:
            self.strategy_function = self.random_strategy
        else:
            self.strategy_function = strategy_function
        self.fitness = 0

    def play(self, history):
        return self.strategy_function(history)

    def calculate_fitness(self, opponent_strategy, num_rounds=100):
        from play_game import play_game # avoid circular import
        score1, score2, history1, history2 = play_game(self, opponent_strategy, num_rounds)
        self.fitness = score1  # Update fitness

    def mutate(self, mutation_rate):
        if random.random() < mutation_rate:
            # Example: Change the strategy function
            self.strategy_function = random.choice([self.always_cooperate, self.always_defect, self.tit_for_tat, self.random_strategy])

    # Example strategies
    def always_cooperate(self, history):
        return 'C'

    def always_defect(self, history):
        return 'D'

    def tit_for_tat(self, history):
        if not history:
            return 'C'
        else:
            return history[-1][1]  # Cooperate if opponent cooperated last round

    def random_strategy(self, history):
        return random.choice(['C', 'D'])


# Tournament selection for evolutionary algorithm
def tournament_selection(population, tournament_size=5):
    """
    Selects individuals using tournament selection.

    Args:
        population: List of Strategy objects
        tournament_size: Number of individuals to compete in each tournament

    Returns:
        The winner of the tournament (Strategy with highest fitness)
    """
    # Select random individuals for the tournament
    tournament = random.sample(population, min(tournament_size, len(population)))
    # Return the individual with the highest fitness
    return max(tournament, key=lambda x: x.fitness)


# Evaluate fitness of all strategies in the population
def evaluate_population(population, opponent_strategies, rounds_per_match=100):
    """
    Evaluate the fitness of each strategy in the population by playing against opponent strategies.

    Args:
        population: List of Strategy objects
        opponent_strategies: List of opponent strategies to play against
        rounds_per_match: Number of rounds per match
    """
    for strategy in population:
        total_score = 0
        for opponent in opponent_strategies:
            score1, _, _, _ = play_game.play_game(strategy, opponent, rounds_per_match)
            total_score += score1
        strategy.fitness = total_score / len(opponent_strategies)


# Crossover between two parent strategies
def crossover(parent1, parent2, crossover_rate=0.7):
    """
    Perform crossover between two parent strategies.

    Args:
        parent1, parent2: Parent Strategy objects
        crossover_rate: Probability of crossover occurring

    Returns:
        Two new Strategy objects (offspring)
    """
    if random.random() < crossover_rate:
        # Simple strategy swap
        strategy_pool = [parent1.strategy_function, parent2.strategy_function]
        child1 = Strategy(random.choice(strategy_pool))
        child2 = Strategy(random.choice(strategy_pool))
        return child1, child2
    else:
        # Clone parents
        child1 = Strategy(parent1.strategy_function)
        child2 = Strategy(parent2.strategy_function)
        return child1, child2


# Complete the evolve function
def evolve(population_size=75, generations=50, tournament_size=5, crossover_rate=0.7, mutation_rate=0.1):
    """
    Run the evolutionary algorithm for the Prisoner's Dilemma.

    Args:
        population_size: Size of the population
        generations: Number of generations to run
        tournament_size: Size of tournament for selection
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation

    Returns:
        The final population of strategies
    """
    # Create initial population
    population = initialize_population(population_size)

    # Create a mix of opponent strategies to test against
    opponent_strategies = [
        Strategy(lambda h: 'C'),  # Always cooperate
        Strategy(lambda h: 'D'),  # Always defect
        Strategy(lambda h: 'C' if not h else h[-1][1])  # Tit-for-tat
    ]

    for generation in range(generations):
        # Evaluate fitness
        evaluate_population(population, opponent_strategies)

        # Create new population
        new_population = []

        # Elitism: Keep the best strategy
        best_strategy = max(population, key=lambda x: x.fitness)
        new_population.append(Strategy(best_strategy.strategy_function))

        # Fill the rest with offspring from tournament selection
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)

            child1, child2 = crossover(parent1, parent2, crossover_rate)

            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

        # Print progress
        if generation % 10 == 0:
            avg_fitness = sum(s.fitness for s in population) / len(population)
            print(f"Generation {generation}: Average fitness = {avg_fitness:.2f}")

    return population
