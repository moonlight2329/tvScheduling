# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="GA TV Scheduling", layout="wide")
st.title("Genetic Algorithm — TV Scheduling")

st.markdown("""
Tv Scheduling using Genetic Algorithm.
""")

# ---------- local CSV ----------
default_csv = Path(__file__).parent / "program_ratings.csv"

if default_csv.exists():
    df, programs, hour_cols = load_ratings(str(default_csv))
else:
    st.error("CSV readed.")
    st.stop()
# ---------------------------------------------------------------

st.subheader("Parameters")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Trial 1**")
    co1 = st.slider("CO_R (Trial 1)", 0.0, 0.95, 0.80, 0.01, key="co1")
    mu1 = st.slider("MUT_R (Trial 1)", 0.01, 0.05, 0.02, 0.01, key="mu1")

    st.markdown("**Trial 2**")
    co2 = st.slider("CO_R (Trial 2)", 0.0, 0.95, 0.70, 0.01, key="co2")
    mu2 = st.slider("MUT_R (Trial 2)", 0.01, 0.05, 0.03, 0.01, key="mu2")

with col_b:
    st.markdown("**Trial 3**")
    co3 = st.slider("CO_R (Trial 3)", 0.0, 0.95, 0.60, 0.01, key="co3")
    mu3 = st.slider("MUT_R (Trial 3)", 0.01, 0.05, 0.04, 0.01, key="mu3")

st.markdown("---")
st.subheader("Global GA Settings")
gen = st.number_input("Generations (GEN)", min_value=10, max_value=2000, value=100, step=10)
pop = st.number_input("Population Size (POP)", min_value=10, max_value=500, value=50, step=10)
elit = st.number_input("Elitism Size", min_value=0, max_value=10, value=2, step=1)
tourn = st.number_input("Tournament Size (k)", min_value=2, max_value=10, value=3, step=1)

# -------------------------------
# Auto-run all trials
# -------------------------------
st.markdown("## Results (Auto-Run)")
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
        elitism=int(elit),
        tournament_k=int(tourn),
    )

    schedule = result["best_schedule"]
    score = result["best_score"]
    table = render_schedule_table(schedule, df, hour_cols)

    st.dataframe(table, use_container_width=True)
    st.metric(label="Total Ratings (Fitness)", value=round(score, 4))
    st.caption(f"CO_R = {co:.2f}, MUT_R = {mu:.2f}, GEN = {gen}, POP = {pop}, ELIT = {elit}, TOURN = {tourn}")
    st.markdown("---")
