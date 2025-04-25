
import random
import csv
import math
from pathlib import Path
import copy

class Leg:
    def __init__(self, mode, destination):
        self.mode = mode  # 'fly' or 'drive'
        self.destination = destination

    def __repr__(self):
        #return f"({self.mode}, {self.destination})"
        return f"{self.destination}"

class Chunk:
    def __init__(self, legs=None):
        self.legs = legs if legs else []

    def destinations(self):
        # Don't count the return flight
        return [leg.destination for leg in self.legs if leg.destination != "SEA"]

    def __repr__(self):
        return f"Chunk({'->'.join([l.destination for l in self.legs])})"

class Tour:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks else []
        self._cached_cost = None  


    def all_destinations(self):
        return [d for chunk in self.chunks for d in chunk.destinations()]

    def is_valid(self, all_stadiums):
        visited = set(self.all_destinations())
        return visited == set(all_stadiums)

    def cost(self):
        if self._cached_cost is not None:
            return self._cached_cost

        total = 0
        for chunk in self.chunks:
            # Each chunk always starts from SEA
            current = 'SEA'
            for leg in chunk.legs:
                if leg.mode == 'drive':
                    total += drive_cost(current, leg.destination)
                elif leg.mode == 'fly':
                    total += fly_cost(current, leg.destination)
                current = leg.destination

        self._cached_cost = total
        return total

    def __repr__(self):
        return f"Tour({self.chunks})"

    @classmethod
    def random(cls, stadiums, max_chunk_size=7):
        remaining = random.sample(stadiums, len(stadiums))
        chunks = []

        while remaining:
            chunk_size = random.randint(1, min(max_chunk_size, len(remaining)))
            chunk_stadiums = [remaining.pop(0) for _ in range(chunk_size)]

            legs = []
            # Fly from SEA to start of chunk
            legs.append(Leg('fly', chunk_stadiums[0]))
        
            # Drive within chunk
            for s in chunk_stadiums[1:]:
                legs.append(Leg('drive', s))

            # Fly back to SEA
            legs.append(Leg('fly', 'SEA'))

            chunks.append(Chunk(legs))

        t = cls(chunks)
        assert t.is_valid(stadiums)
        return t

    def hybrid_crossover(self, parent2, lambda_=0.1, max_chunk_size=7):
        def softmax(distances, lambda_):
            weights = {s: math.exp(-lambda_ * d) for s, d in distances.items()}
            total = sum(weights.values())
            return {s: w / total for s, w in weights.items()}

        def sample_from_distribution(probabilities):
            r = random.random()
            cumulative = 0
            for item, p in probabilities.items():
                cumulative += p
                if r <= cumulative:
                    return item
            # Should never get here, but in case of floating point precision wierdness
            return item  

        def choose_next(src, possibilities, lambda_):
            '''Chose among possibilities, favoring ones close to src.'''
            dists = {s: distance_miles[(src, s)] for s in possibilities}
            probs = softmax(dists, lambda_)
            next_stop = sample_from_distribution(probs)
            return next_stop

        stadiums = self.all_destinations()
        used = set()
        inherited_chunks = []

        all_chunks = self.chunks + parent2.chunks
        random.shuffle(all_chunks)

        # Step 1: inherit non-overlapping chunks from parents
        for chunk in all_chunks:
            chunk_stadiums = chunk.destinations()
            if all(s not in used for s in chunk_stadiums):
                inherited_chunks.append(copy.deepcopy(chunk))
                used.update(chunk_stadiums)
            if len(used) >= len(stadiums):
                break

        # Step 2: fill in remaining with softmax sampling
        remaining = list(set(stadiums) - used)
        constructed_chunks = []
        while remaining:
            chunk_size = random.randint(1, min(max_chunk_size, len(remaining)))
            chunk_legs = []

            # Start new chunk: SEA → first destination
            next_stop = choose_next('SEA', remaining, lambda_)
            chunk_legs.append(Leg('fly', next_stop))
            remaining.remove(next_stop)
            current = next_stop

            # Drive within chunk
            while remaining and len(chunk_legs) < chunk_size:
                next_stop = choose_next(current, remaining, lambda_)
                chunk_legs.append(Leg('drive', next_stop))
                remaining.remove(next_stop)
                current = next_stop

            # End chunk with flight back to SEA
            chunk_legs.append(Leg('fly', 'SEA'))
            constructed_chunks.append(Chunk(chunk_legs))

        return Tour(inherited_chunks + constructed_chunks)        

    def mutate(self, mutation_rate=0.3):
        if random.random() > mutation_rate:
            return self  # no mutation

        # Should never have called cost() before mutating, but just in case
        self._cached_cost = None

        # Flatten drive legs with chunk and index info
        drive_legs = []
        for ci, chunk in enumerate(self.chunks):
            for li, leg in enumerate(chunk.legs):
                if leg.mode == 'drive':
                    drive_legs.append((ci, li))

        if len(drive_legs) < 2:
            return self  # nothing to swap

        (ci1, li1), (ci2, li2) = random.sample(drive_legs, 2)

        # Swap the destinations
        d1 = self.chunks[ci1].legs[li1].destination
        d2 = self.chunks[ci2].legs[li2].destination
        self.chunks[ci1].legs[li1].destination = d2
        self.chunks[ci2].legs[li2].destination = d1

        return self


    def mutate_reorder_chunk(tour, mutation_rate=0.3):
        if random.random() > mutation_rate:
            return tour  # No mutation applied

        tour._cached_cost = None  # Clear cached cost

        # Only consider chunks with at least 2 swappable legs (excluding final return flight)
        candidate_chunks = [chunk for chunk in tour.chunks if len(chunk.legs) > 2]
        if not candidate_chunks:
            return tour

        chunk = random.choice(candidate_chunks)

        # Valid swap indices: all except the last leg (which is always 'fly' to 'SEA')
        swappable_indices = list(range(len(chunk.legs) - 1))
        if len(swappable_indices) < 2:
            return tour

        i1, i2 = random.sample(swappable_indices, 2)

        # Swap the destinations
        d1 = chunk.legs[i1].destination
        d2 = chunk.legs[i2].destination
        chunk.legs[i1].destination = d2
        chunk.legs[i2].destination = d1

        return tour


    def mutate_split(tour, mutation_rate=0.1):
        if random.random() > mutation_rate:
            return tour  # no mutation

        tour._cached_cost = None

        candidate_chunks = [i for i, chunk in enumerate(tour.chunks) if len(chunk.destinations()) >= 2]
        if not candidate_chunks:
            return tour

        ci = random.choice(candidate_chunks)
        chunk = tour.chunks.pop(ci)
        dests = chunk.destinations()

        split_point = random.randint(1, len(dests) - 1)
        first, second = dests[:split_point], dests[split_point:]

        def build_chunk(stadiums):
            legs = [Leg('fly', stadiums[0])]
            for s in stadiums[1:]:
                legs.append(Leg('drive', s))
            legs.append(Leg('fly', 'SEA'))
            return Chunk(legs)

        tour.chunks.insert(ci, build_chunk(second))
        tour.chunks.insert(ci, build_chunk(first))

        return tour
 

    def mutate_merge(tour, max_chunk_size=7, mutation_rate=0.1):
        if random.random() > mutation_rate:
            return tour  # no mutation
        
        tour._cached_cost = None

        mergeable_pairs = []
        for i in range(len(tour.chunks) - 1):
            size = len(tour.chunks[i].destinations()) + len(tour.chunks[i + 1].destinations())
            if size <= max_chunk_size:
                mergeable_pairs.append(i)

        if not mergeable_pairs:
            return tour

        i = random.choice(mergeable_pairs)
        c1 = tour.chunks.pop(i)
        c2 = tour.chunks.pop(i)  # same index, after popping c1

        merged_dests = c1.destinations() + c2.destinations()
        legs = [Leg('fly', merged_dests[0])]
        for s in merged_dests[1:]:
            legs.append(Leg('drive', s))
        legs.append(Leg('fly', 'SEA'))

        tour.chunks.insert(i, Chunk(legs))

        return tour



# This will be populated at import time
distance_miles = {}

# Load distances from CSV once at module load
def load_distances(filename='all_pairs_distance.csv'):
    path = Path(__file__).parent / filename
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            origin = row['origin']
            dest = row['destination']
            dist = float(row['dist_mi'])
            distance_miles[(origin, dest)] = dist
            distance_miles[(dest, origin)] = dist  # add reverse direction
    print(f"Loaded {len(distance_miles)} distances.")

# Call it at module load
load_distances()


def drive_cost(origin, destination):
    key = (origin, destination)
    assert key in distance_miles
    miles = distance_miles.get(key, float('inf'))
    avg_mph = 50
    time_hours = miles / avg_mph
    if time_hours <= 3:
        drive_time_cost = 0
    else:
        """
        > floor((((1:24)-3)*10)^1.5)
        [1]  NaN  NaN    0   31   89  164  252  353  464  585  715  853 1000 1153 1314 1482 1656 1837 2023 2216 2414 2618 2828
        [24] 3043
        """
        drive_time_cost_base = 10
        drive_time_cost_exp = 1.5
        drive_time_cost = ((time_hours-3) * drive_time_cost_base) ** drive_time_cost_exp
       
    rental_car_cost_per_day = 80
    gas_cost_per_mile = 0.11

    return drive_time_cost + rental_car_cost_per_day + miles * gas_cost_per_mile


# Maps a team to their one-way flight cost from SEA
flight_cost_from_sea = {}

def load_airports(filename='mlb_airports.csv'):
    path = Path(__file__).parent / filename
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            team = row['team']
            cost = float(row['one_way_cost'])
            flight_cost_from_sea[team] = cost
    print(f"Loaded SEA flight costs for {len(flight_cost_from_sea)} teams.")

load_airports()

def fly_cost(origin, destination, n_pax=2):
    assert origin == "SEA" or destination == "SEA"
    if origin == 'SEA':
        assert destination in flight_cost_from_sea
        return n_pax * flight_cost_from_sea[destination]
    else:
        assert origin in flight_cost_from_sea
        return n_pax * flight_cost_from_sea[origin]

