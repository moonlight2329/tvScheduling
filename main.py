import streamlit as st
import csv
import random
import io
from urllib.parse import urlparse

st.set_page_config(page_title="TV Scheduling (Genetic Algorithm)", layout="wide")

# ------------------------------- CSV READER --------------------------------- #
@st.cache_data
def read_csv_to_dict(file_path: str):
    """
    Read a CSV either from a local path or a URL and return:
      - program_ratings: dict[str, list[float]]
      - time_labels: list[str] (inferred from header if present; otherwise None)
    Expected CSV:
       Program, 06:00, 07:00, ..., 23:00
       P1,      1.2,  3.4,   ...,  2.1
       ...
    """
    program_ratings = {}

    def _parse_text(text: str):
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)  # header row
        for row in reader:
            if not row:
                continue
            program = row[0].strip()
            ratings = []
            for x in row[1:]:
                try:
                    ratings.append(float(x))
                except (ValueError, TypeError):
                    ratings.append(0.0)
            program_ratings[program] = ratings

        # Infer time labels from header (skip first column which is Program)
        time_labels = None
        if header and len(header) > 1:
            time_labels = [h.strip() or f"Slot {i}" for i, h in enumerate(header[1:], start=1)]
        return program_ratings, time_labels

    parsed = urlparse(file_path or "")
    if parsed.scheme in ("http", "https"):
        try:
            from urllib.request import urlopen
            with urlopen(file_path) as resp:
                text = resp.read().decode("utf-8")
            return _parse_text(text)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch CSV from URL: {e}")
    else:
        try:
            with open(file_path, mode="r", newline="", encoding="utf-8") as f:
                text = f.read()
            return _parse_text(text)
        except Exception as e:
            raise RuntimeError(f"Failed to read local CSV: {e}")

# ----------------------------- DEFAULT DATA SOURCE -------------------------- #
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/moonlight2329/tvScheduling/main/program_ratings.csv"

st.sidebar.header("Data Source")
csv_path = st.sidebar.text_input("CSV URL or local path", value=DEFAULT_CSV_URL)

try:
    program_ratings_dict, inferred_time_labels = read_csv_to_dict(csv_path)
except Exception as err:
    st.error(str(err))
    st.stop()

if not program_ratings_dict:
    st.error("The CSV appears empty or malformed (no rows after header).")
    st.stop()

# ------------------------------ TIME LABELS --------------------------------- #
# If header has time labels, use them; otherwise default to 06:00..(n slots)
first_len = len(next(iter(program_ratings_dict.values())))
if inferred_time_labels and len(inferred_time_labels) == first_len:
    time_labels = inferred_time_labels
else:
    # Default synthetic labels if header not present or mismatched
    time_labels = [f"{(6 + i) % 24:02d}:00" for i in range(first_len)]
num_slots = len(time_labels)

# Validate columns length across rows
lens = {len(v) for v in program_ratings_dict.values()}
if len(lens) != 1:
    st.error("Inconsistent number of rating columns across programs in the CSV.")
    st.stop()
if lens.pop() != num_slots:
    st.error("Internal shape mismatch: header/labels and data columns differ.")
    st.stop()

all_programs = list(program_ratings_dict.keys())
ratings = program_ratings_dict

# --------------------------- GA PARAMS & OPTIONS ---------------------------- #
st.sidebar.header("Genetic Algorithm Parameters")
GEN = st.sidebar.slider("Generations", 10, 500, 120)
POP = st.sidebar.slider("Population Size", 10, 300, 100)
CO_R = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.85)
MUT_R = st.sidebar.slider("Mutation Rate", 0.01, 1.0, 0.2)
EL_S = st.sidebar.slider("Elitism Size", 0, 20, 4)

st.sidebar.header("Scheduling Rules")
default_allow_repeats = len(all_programs) < num_slots
ALLOW_REPEATS = st.sidebar.checkbox("Allow program repeats across time slots", value=default_allow_repeats)
NO_CONSEC_DUP = st.sidebar.checkbox("Disallow consecutive duplicate programs (soft penalty)", value=True)

if not ALLOW_REPEATS and len(all_programs) < num_slots:
    st.error(
        f"Fewer programs ({len(all_programs)}) than time slots ({num_slots}). "
        "Enable 'Allow program repeats' or add more programs."
    )
    st.stop()

# ------------------------------- GA HELPERS --------------------------------- #
def fitness_function(schedule):
    """
    Sum the ratings for the chosen (program, time_slot) pairs.
    Apply a penalty if NO_CONSEC_DUP is enabled and consecutive slots repeat.
    """
    total = 0.0
    for slot_idx, program in enumerate(schedule):
        if program not in ratings:
            return 0.0
        total += ratings[program][slot_idx]

    if NO_CONSEC_DUP:
        # Penalize consecutive duplicates to encourage variety
        penalty = 0
        for i in range(1, len(schedule)):
            if schedule[i] == schedule[i - 1]:
                penalty += 1
        # Each consecutive repeat reduces total a bit (tune as needed)
        total -= penalty * 0.01  # small penalty per consecutive repeat
    return total

def make_random_schedule():
    if ALLOW_REPEATS:
        return [random.choice(all_programs) for _ in range(num_slots)]
    else:
        # unique assignment
        return random.sample(all_programs, num_slots)

def crossover(schedule1, schedule2):
    """
    One-point crossover. If repeats are not allowed, repair to ensure uniqueness.
    """
    if len(schedule1) < 2:
        return schedule1[:], schedule2[:]
    cp = random.randint(1, len(schedule1) - 1)
    child1 = schedule1[:cp] + schedule2[cp:]
    child2 = schedule2[:cp] + schedule1[cp:]

    if not ALLOW_REPEATS:
        # Repair to enforce uniqueness
        def repair_unique(child):
            seen = set()
            new_child = []
            for p in child:
                if p not in seen:
                    new_child.append(p)
                    seen.add(p)
            missing = [p for p in all_programs if p not in seen]
            random.shuffle(missing)
            new_child.extend(missing)
            return new_child[:num_slots]
        child1 = repair_unique(child1)
        child2 = repair_unique(child2)
    return child1, child2

def mutate(schedule):
    """
    Swap two positions (works for both repeat and unique modes).
    Additionally, with small chance, replace one gene (only if repeats allowed).
    """
    if len(schedule) >= 2:
        i, j = random.sample(range(len(schedule)), 2)
        schedule[i], schedule[j] = schedule[j], schedule[i]
    if ALLOW_REPEATS and random.random() < 0.2:
        k = random.randrange(len(schedule))
        schedule[k] = random.choice(all_programs)
    return schedule

def genetic_algorithm(generations, population_size, crossover_rate, mutation_rate, elitism_size):
    # Initialize population
    population = [make_random_schedule() for _ in range(population_size)]

    progress_bar = st.progress(0)
    for g in range(generations):
        # Sort by fitness (desc)
        population.sort(key=fitness_function, reverse=True)

        # Elitism
        new_population = population[:elitism_size]

        # Fill the rest
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:max(2, population_size // 2)], 2)  # bias to fitter half
            if random.random() < crossover_rate:
                c1, c2 = crossover(parent1, parent2)
            else:
                c1, c2 = parent1[:], parent2[:]

            if random.random() < mutation_rate:
                c1 = mutate(c1)
            if random.random() < mutation_rate:
                c2 = mutate(c2)

            new_population.extend([c1, c2])

        population = new_population[:population_size]
        progress_bar.progress((g + 1) / generations)

    progress_bar.empty()
    population.sort(key=fitness_function, reverse=True)
    return population[0]

# ------------------------------- RUN GA ------------------------------------- #
best_schedule = genetic_algorithm(
    generations=GEN,
    population_size=POP,
    crossover_rate=CO_R,
    mutation_rate=MUT_R,
    elitism_size=EL_S,
)

# ----------------------------- SHOW RESULTS --------------------------------- #
st.header("Optimal Schedule (Genetic Algorithm)")

rows = [(time_labels[i], best_schedule[i]) for i in range(num_slots)]
st.table(rows)
st.metric("Total Ratings", f"{fitness_function(best_schedule):.4f}")

with st.expander("Parameters & Data Summary"):
    st.write(
        {
            "Generations": GEN,
            "Population Size": POP,
            "Crossover Rate": CO_R,
            "Mutation Rate": MUT_R,
            "Elitism Size": EL_S,
            "CSV Source": csv_path,
            "Programs Count": len(all_programs),
            "Time Slots": num_slots,
            "Allow Repeats": ALLOW_REPEATS,
            "No Consecutive Duplicates (penalized)": NO_CONSEC_DUP,
        }
    )

st.caption("Tip: To avoid the earlier error, enable 'Allow program repeats' when programs < time slots, or add more programs in the CSV.")
