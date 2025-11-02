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
      - header: list[str] or None (the header row that was skipped)
    Expected CSV:
       Program, 06:00, 07:00, ..., 23:00
       P1,      1.2,  3.4,   ...,  2.1
       ...
    """
    program_ratings = {}

    def _parse_text(text: str):
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)  # skip header but return for info
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
        return program_ratings, header

    parsed = urlparse(file_path or "")
    if parsed.scheme in ("http", "https"):
        # Read from URL (works with raw.githubusercontent.com)
        try:
            from urllib.request import urlopen
            with urlopen(file_path) as resp:
                text = resp.read().decode("utf-8")
            return _parse_text(text)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch CSV from URL: {e}")
    else:
        # Read from local file
        try:
            with open(file_path, mode="r", newline="", encoding="utf-8") as f:
                text = f.read()
            return _parse_text(text)
        except Exception as e:
            raise RuntimeError(f"Failed to read local CSV: {e}")

# ----------------------------- DEFAULT DATA SOURCE -------------------------- #
# ✅ Corrected "raw" GitHub URL (remove 'refs/heads/')
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/moonlight2329/tvScheduling/main/program_ratings.csv"

# UI: allow changing the CSV source if you want
st.sidebar.header("Data Source")
csv_path = st.sidebar.text_input("CSV URL or local path", value=DEFAULT_CSV_URL)

# Try reading the CSV
try:
    program_ratings_dict, header_row = read_csv_to_dict(csv_path)
except Exception as err:
    st.error(str(err))
    st.stop()

# Show raw data (optional)
with st.expander("Show parsed Program Ratings (dict)"):
    st.write(program_ratings_dict)

# --------------------------- GA PARAMS & TIME SLOTS ------------------------- #
st.sidebar.header("Genetic Algorithm Parameters")
GEN = st.sidebar.slider("Generations", 10, 500, 100)
POP = st.sidebar.slider("Population Size", 10, 200, 50)
CO_R = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)
MUT_R = st.sidebar.slider("Mutation Rate", 0.01, 1.0, 0.2)
EL_S = st.sidebar.slider("Elitism Size", 0, 10, 2)

# Time slots: 06:00 .. 23:00 (18 slots)
all_time_slots = list(range(6, 24))
num_slots = len(all_time_slots)

# Basic validations
if not program_ratings_dict:
    st.error("The CSV appears empty or malformed (no rows after header).")
    st.stop()

# Ensure every program has the same number of ratings and matches the time slots
lens = {len(v) for v in program_ratings_dict.values()}
if len(lens) != 1:
    st.error("Inconsistent number of rating columns across programs in the CSV.")
    st.stop()

ratings_len = lens.pop()
if ratings_len != num_slots:
    st.error(
        f"Error: Number of time slots ({num_slots}) does not match "
        f"ratings per program in CSV ({ratings_len})."
    )
    st.stop()

all_programs = list(program_ratings_dict.keys())

if len(all_programs) < num_slots:
    st.error(
        f"Error: Fewer programs ({len(all_programs)}) than time slots ({num_slots}). "
        "Add more programs or reduce the number of time slots."
    )
    st.stop()

# We'll use all programs; schedules will be length = num_slots chosen from all_programs
ratings = program_ratings_dict  # alias for fitness function

# ------------------------------- GA HELPERS --------------------------------- #
def fitness_function(schedule):
    """
    Sum the ratings for the chosen (program, time_slot) pairs.
    schedule is a list of program names of length num_slots.
    """
    total_rating = 0.0
    for time_slot_idx, program in enumerate(schedule):
        # Defensive checks
        if program not in ratings:
            return 0.0
        if time_slot_idx >= len(ratings[program]):
            return 0.0
        total_rating += ratings[program][time_slot_idx]
    return total_rating

def crossover(schedule1, schedule2):
    """
    One-point crossover, then repair to keep unique programs and correct length.
    """
    if len(schedule1) < 2:
        return schedule1[:], schedule2[:]
    crossover_p_
