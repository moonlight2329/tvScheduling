
# GA TV Scheduling — Streamlit App

This app implements a **Genetic Algorithm (GA)** to build a TV program schedule across hourly slots (6:00–23:00) using a ratings matrix.

## Files
- `app.py` — Streamlit app
- `program_ratings.csv` — Ratings input (Program vs Hour 6..Hour 23)
- `requirements.txt` — Minimal dependencies

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL shown in the terminal.

## How to deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repository**.
2. On https://streamlit.io/cloud , choose **New app** → point to your repo and `app.py`.
3. Set Python version if needed, and deploy.

## CSV format
The CSV must include the first column named `Type of Program` and hour columns:
```
Type of Program,Hour 6,Hour 7,...,Hour 23
news,0.1,0.1,...,0.2
live_soccer,0.0,0.0,...,0.3
...
```
You can modify rating values per the assignment and redeploy.

## Notes
- Schedule allows **repeats** (a program may appear in multiple hours).
- Fitness = sum of selected program ratings across all hour slots.
- You can vary **CO_R** (0.0–0.95) and **MUT_R** (0.01–0.05) per trial, plus set generations, population size, elitism, and tournament size.
- The app runs **three trials** and shows each schedule as a table with its total rating.
