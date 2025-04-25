from tour import Tour
import random

def initialize_population(stadiums, population_size=100):
    return [Tour.random(stadiums) for _ in range(population_size)]


def tournament_selection(population, k=5):
    '''Choose the best from a simple random sample of size k'''
    selected = random.sample(population, k)
    selected.sort(key=lambda t: t.cost())
    return selected[0]


def genetic_algorithm(stadiums, generations=500, pop_size=100):
    population = initialize_population(stadiums, pop_size)
    best = min(population, key=lambda t: t.cost())
    
    for gen in range(generations):
        #print(f"Generation {gen}")

        # Evaluate all individuals
        population.sort(key=lambda tour: tour.cost())

        # Elitism: preserve the top N
        elite_count = 1
        next_generation = population[:elite_count]

        # Fill in the rest with offspring
        while len(next_generation) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = p1.hybrid_crossover(p2)
            child = child.mutate()
            child = child.mutate_reorder_chunk()
            child = child.mutate_split()
            child = child.mutate_merge()
            # Paranoia
            assert child.is_valid(stadiums)
            next_generation.append(child)
        
        generation_best = min(next_generation, key=lambda t: t.cost())
        if generation_best.cost() < best.cost():
            best = generation_best
            print(f"Generation {gen} new best! {best.cost()}, {best}")

        population = next_generation

        # Inject new random individuals every so often
        if gen % 20 == 0:
            for i in range(int(0.05 * pop_size), pop_size):
                population[i] = Tour.random(stadiums)

    
    return best, best.cost()


def main():
    # SEA missing as that's home, can visit for free
    stadiums = ["ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET", 
                "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
                "PHI", "PIT", "SDP", "SFG", "STL", "TBR", "TEX", "TOR", "WAS"]


    (best, best_cost) = genetic_algorithm(stadiums, generations=5000, pop_size=200)


if __name__ == '__main__':
    main()
