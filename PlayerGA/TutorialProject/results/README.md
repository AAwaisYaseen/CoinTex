# COMP6009 Assignment A Hazard-Aware A* for CoinTex

Ahmed Awais Yaseen · 22904430 · Group 13

## Files

| File | Description |
|---|---|
| `main_work_astar.py` | Proposed Hazard-Aware A* controller |
| `main_work_greedy.py` | Supplied Greedy controller, with result logging and a 30 s timeout added |
| `main_updated.py` | Lab 2 Genetic Algorithm, with the same logging and timeout added |
| `analyse_results.py` | Creates the charts and summary tables used in the report |

## How to run

1. Clone CoinTex and install its requirements:
   ```bash
   git clone https://github.com/ahmedfgad/CoinTex.git
   cd CoinTex/PlayerGA
   pip install -r requirements.txt
   pip install pandas matplotlib
   ```
2. Copy the files above into `CoinTex/PlayerGA/TutorialProject`.
3. Run one controller from that folder, then select a level in the game:
   ```bash
   cd TutorialProject
   python3 main_work_astar.py     # or main_work_greedy.py / main_updated.py
   ```
   Each run is saved to `results/test_runs.csv`.
4. To recreate the charts, place the result CSVs from the GitHub repository in `TutorialProject/results/` and run:
   ```bash
   python3 analyse_results.py
   ```
   Output is saved to `results/analysis/`.

## Results

Result files are in the GitHub repository: **<https://github.com/AAwaisYaseen/CoinTex>**

- `results/final_comparison_results.csv` - 240 runs (3 algorithms × 8 levels × 10 runs)
- `results/astar_optimisation_pilot.csv` – 50 runs (initial vs optimised A*)
- `results/analysis/` – charts and summary tables
