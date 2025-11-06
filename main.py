import streamlit as st
import pandas as pd
import random
from typing import List, Tuple, Dict
from pathlib import Path

# ---------------------------------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ---------------------------------------------------------
st.set_page_config(page_title="GA TV Scheduling", layout="wide")

# =========================
# Data loading
# =========================
@st.cache_data
def load_ratings(csv_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = pd.read_csv(csv_path)
    if "Type of Program" not in df.columns:
        raise ValueError("CSV must include a 'Type of Program' column.")
    hour_cols = [c for c in df.columns if c != "Type of Program"]
    programs = df["Type of Program"].tolist()
    return df, programs, hour_cols


def fitness(schedule: List[str], df: pd.DataFrame, hour_cols: List[str]) -> float:
    program_to_row = {p: i for i, p in enumerate(df["Type of Program"])}
    total = 0.0
    for idx, program in enumerate(schedule):
        row_i = program_to_row[program]
        hour_col = hour_cols[idx]
        total += float(df.at[row_i, hour_col])
    return total


def random_schedule(programs: List[str], num_hours: int) -> List[str]:
    return [random.choice(programs) for _ in range(num_hours)]


def single_point_crossover(p1: List[str], p2: List[str]):
    if len(p1) < 2:
        return p1[:], p2[:]
    cut = random.randint(1, len(p1) - 1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]


def mutate(schedule: List[str], programs: List[str]) -> List[str]:
    i = random.randrange(len(schedule))
    schedule[i] = random.choice(programs)
    return schedule


def tournament_selection(pop, df, hour_cols, k=3):
    candidates = random.sample(pop, k=min(k, len(pop)))
    candidates.sort(key=lambda s: fitness(s, df, hour_cols), reverse=True)
    return candidates[0]


def run_ga(
    df: pd.DataFrame,
    programs: List[str],
    hour_cols: List[str],
    generations: int = 100,
    pop_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
) -> Dict:

    num_hours = len(hour_cols)
    population = [random_schedule(programs, num_hours) for _ in range(pop_size)]
    best = max(population, key=lambda s: fitness(s, df, hour_cols))
    best_score = fitness(best, df, hour_cols)

    for _ in range(generations):
        new_pop = []

        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, df, hour_cols)
            p2 = tournament_selection(population, df, hour_cols)

            if random.random() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            if random.random() < mutation_rate:
                c1 = mutate(c1, programs)
            if random.random() < mutation_rate:
                c2 = mutate(c2, programs)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        gen_best = max(population, key=lambda s: fitness(s, df, hour_cols))
        gen_score = fitness(gen_best, df, hour_cols)

        if gen_score > best_score:
            best = gen_best
            best_score = gen_score

    return {
        "best_schedule": best,
        "best_score": best_score,
    }


def render_schedule_table(schedule, df, hour_cols):
    program_to_row = {p: i for i, p in enumerate(df["Type of Program"])}
    rows = []
    for idx, program in enumerate(schedule):
        hour_label = hour_cols[idx]
        hour_number = hour_label.split()[-1]
        rating = float(df.at[program_to_row[program], hour_label])
        rows.append(
            {"Hour": f"{hour_number}:00", "Program": program, "Rating": rating}
        )
    return pd.DataFrame(rows)


# =========================
# STREAMLIT PAGE
# =========================
st.title("Genetic Algorithm — TV Scheduling")
st.markdown("TV Scheduling using Genetic Algorithm.")

# Load CSV
default_csv = Path(__file__).parent / "program_ratings.csv"

if default_csv.exists():
    df, programs, hour_cols = load_ratings(str(default_csv))
else:
    st.error("CSV not found.")
    st.stop()

# ======== Parameters ========
st.subheader("GA Parameters")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Trial 1**")
    co1 = st.slider("CO_R (Trial 1)", 0.0, 0.95, 0.10, 0.01)
    mu1 = st.slider("MUT_R (Trial 1)", 0.01, 0.05, 0.04, 0.01)

    st.markdown("**Trial 2**")
    co2 = st.slider("CO_R (Trial 2)", 0.0, 0.95, 0.20, 0.01)
    mu2 = st.slider("MUT_R (Trial 2)", 0.01, 0.05, 0.04, 0.01)

with col_b:
    st.markdown("**Trial 3**")
    co3 = st.slider("CO_R (Trial 3)", 0.0, 0.95, 0.30, 0.01)
    mu3 = st.slider("MUT_R (Trial 3)", 0.01, 0.05, 0.05, 0.01)

st.markdown("---")
st.subheader("Global GA Settings")
gen = int(st.text_input("Generations (GEN)", "100"))
pop = int(st.text_input("Population Size (POP)", "50"))

# ========== Auto-run Trials ==========
st.markdown("## Results of the trial")

trials = [
    ("Trial 1", co1, mu1),
    ("Trial 2", co2, mu2),
    ("Trial 3", co3, mu3),
]

for label, co, mu in trials:
    st.markdown(f"### {label}")

    result = run_ga(
        df, programs, hour_cols,
        generations=int(gen),
        pop_size=int(pop),
        crossover_rate=float(co),
        mutation_rate=float(mu),
    )

    schedule = result["best_schedule"]
    score = result["best_score"]
    table = render_schedule_table(schedule, df, hour_cols)

    st.dataframe(table, use_container_width=True)
    st.metric(label="Total Ratings (Fitness)", value=round(score, 4))
    st.caption(
        f"CO_R = {co:.2f}, MUT_R = {mu:.2f}, GEN = {gen}, POP = {pop}"
    )
    st.markdown("---")
