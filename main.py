import streamlit as st
import csv
import random

# Function to read the CSV file and convert it to the desired format
@st.cache_data # Cache the data reading for better performance
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)

        for row in reader:
            program = row[0]
            # Safely convert ratings to floats, handling potential errors
            ratings = []
            for x in row[1:]:
                try:
                    ratings.append(float(x))
                except ValueError:
                    ratings.append(0.0) # Handle non-numeric values

            program_ratings[program] = ratings

    return program_ratings

# Path to the CSV file (assuming it's in the same directory as the Streamlit app)
file_path = 'https://raw.githubusercontent.com/moonlight2329/tvScheduling/refs/heads/main/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

# Display the raw data (optional)
st.write("Program Ratings Data:")
st.write(program_ratings_dict)


##################################### DEFINING PARAMETERS AND DATASET ################################################################
# Sample rating programs dataset for each time slot.
ratings = program_ratings_dict

# Streamlit inputs for parameters (optional, for user interaction)
st.sidebar.header("Genetic Algorithm Parameters")
GEN = st.sidebar.slider("Generations", 10, 500, 100)
POP = st.sidebar.slider("Population Size", 10, 200, 50)
CO_R = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)
MUT_R = st.sidebar.slider("Mutation Rate", 0.01, 1.0, 0.2)
EL_S = st.sidebar.slider("Elitism Size", 0, 10, 2)


all_programs = list(ratings.keys()) # all programs
all_time_slots = list(range(6, 24)) # time slots

# Ensure the number of time slots matches the number of ratings per program
if len(all_time_slots) != len(list(ratings.values())[0]):
    st.error("Error: The number of time slots does not match the number of ratings per program in the CSV.")
    st.stop()


######################################### DEFINING FUNCTIONS ########################################################################
# defining fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        # Ensure the program exists in ratings and the time_slot index is valid
        if program in ratings and time_slot < len(ratings[program]):
             total_rating += ratings[program][time_slot]
        else:
            # Handle cases where program or time slot is invalid in the schedule
            return 0 # Or a very low penalty value
    return total_rating

# initializing the population (Brute force - use only for small datasets)
@st.cache_data
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    # Using itertools.permutations for a more robust permutation generation
    import itertools
    for schedule in itertools.permutations(programs, len(time_slots)):
         all_schedules.append(list(schedule))

    return all_schedules

# selection (Brute force - use only for small datasets)
@st.cache_data
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = -1  # Initialize with a value lower than any possible rating

    # Add a progress bar for potentially long calculations
    progress_bar = st.progress(0)
    total_schedules = len(all_schedules)

    for i, schedule in enumerate(all_schedules):
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

        # Update progress bar
        progress_bar.progress((i + 1) / total_schedules)

    progress_bar.empty() # Remove progress bar when done
    return best_schedule

# calling the pop func. (Commented out as brute force is likely too slow for larger datasets)
# all_possible_schedules = initialize_pop(all_programs, all_time_slots)

# callin the schedule func. (Commented out as brute force is likely too slow for larger datasets)
# best_schedule_brute_force = finding_best_schedule(all_possible_schedules)


############################################# GENETIC ALGORITHM #############################################################################

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]

    # Simple repair mechanism to ensure each program appears only once
    def repair(child):
        seen = set()
        new_child = []
        for program in child:
            if program not in seen:
                new_child.append(program)
                seen.add(program)
        # Fill in missing programs
        missing = [p for p in all_programs if p not in seen]
        random.shuffle(missing)
        new_child.extend(missing)
        return new_child[:len(all_time_slots)] # Ensure correct length

    return repair(child1), repair(child2)


# mutating
def mutate(schedule):
    mutation_point1, mutation_point2 = random.sample(range(len(schedule)), 2)
    schedule[mutation_point1], schedule[mutation_point2] = schedule[mutation_point2], schedule[mutation_point1] # Swap mutation
    return schedule

# calling the fitness func.
def evaluate_fitness(schedule):
    return fitness_function(schedule)

# genetic algorithms with parameters
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):

    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    # Add a progress bar for the genetic algorithm
    progress_bar = st.progress(0)

    for generation in range(generations):
        new_population = []

        # Elitsm
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population
        progress_bar.progress((generation + 1) / generations) # Update progress bar

    progress_bar.empty() # Remove progress bar when done
    return population[0]

##################################################### RESULTS ###################################################################################

# Use the genetic algorithm to find a good schedule
# Start with a random initial schedule since brute force is commented out
initial_schedule_ga = random.sample(all_programs, len(all_time_slots))

genetic_schedule = genetic_algorithm(initial_schedule_ga, generations=GEN, population_size=POP, elitism_size=EL_S)

# Display the results using Streamlit
st.header("Optimal Schedule (Genetic Algorithm)")

# Create a list of tuples for displaying the schedule
schedule_data = []
for time_slot, program in zip(all_time_slots, genetic_schedule):
    schedule_data.append((f"{time_slot:02d}:00", program))

# Display the schedule as a table
st.table(schedule_data)

st.write("Total Ratings:", fitness_function(genetic_schedule))

# You can add more Streamlit components here, e.g., charts, explanations, etc.
