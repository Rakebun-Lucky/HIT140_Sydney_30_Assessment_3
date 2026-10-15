# HIT140 Sydney Group 30 – Assessment 3

Multiple linear regression on the 104 matches of the **2026 FIFA World Cup** (72 group-stage and 32 knockout matches). The project asks how much of a match's outcome can be explained by information that is known **before kick-off**.

| Model | Response variable | Rows | Notebook |
|---|---|---|---|
| **LR 2.1** | Goal difference of a match (home − away) | 104 matches | `linear_regression_2.1/LR2_1_part1_data_eda.ipynb` |
| **LR 2.2** | Goals scored by one team in one match | 208 team-match rows (104 × 2) | `linear_regression_2.2/LR2_2_goals_scored.ipynb` |
| **LR 2.2.3** (extension) | Goals scored, with a player-level predictor added | 208 rows | `linear_regression_2.2/LR2_2_3_goal_deference.ipynb` |

## Team and contributions

| Member | Contribution |
|---|---|
| Rakebun Islam | LR 2.1 Part 1 – data checking, feature engineering, EDA |
| Saimon Dahal | LR 2.1 Part 2 – modelling and evaluation |
| Nabin Pandey | LR 2.2 – goals scored (data, EDA, modelling, evaluation) |
| Fuad | LR 2.2.3 – extension with player data *(please confirm)* |

## Repository structure

```
.
├── README.md
├── pyproject.toml
├── data/
│   ├── matches_clean.csv         104 matches, 21 columns
│   ├── team_matches_clean.csv    208 team-match rows, 18 columns
│   ├── team_info.csv             48 teams: FIFA rank, pot, World Cup appearances, confederation
│   ├── players_clean.csv         1,039 players, 67 columns (FBref-style column names)
│   ├── model_2_1.csv             modelling table for LR 2.1 (104 × 12)
│   └── model_2_2.csv             modelling table for LR 2.2 (208 × 13)
├── linear_regression_2.1/
│   └── LR2_1_part1_data_eda.ipynb          Part 1 (features, EDA) and Part 2 (models)
└── linear_regression_2.2/
    ├── LR2_2_goals_scored.ipynb
    ├── LR2_2_3_goal_deference.ipynb
    └── figures/
        ├── fig_correlation.png
        ├── fig_actual_vs_predicted.png
        ├── fig_diagnostics_model_1.png
        ├── fig_diagnostics_model_2.png
        ├── results_table.csv
        └── coefficients_model2.csv
```

## Method in brief

- **No look-ahead.** Form, points per game and rest days use only matches played *before* the match being predicted; a match is added to a team's history only after its features are built. A team's first match starts from the tournament-average form rather than zero.
- **Predictors.** All LR 2.1 predictors are home-minus-away differences (FIFA rank, World Cup appearances, host status, rest days, form, points per game, squad age, strong-confederation flag). LR 2.2 uses rank difference, host advantage, strong-confederation difference, kick-off hour, attack form, opponent defence form, knockout indicator and a high-altitude-venue flag.
- **Multicollinearity.** Checked with variance inflation factors (all below 3 in LR 2.1 and below 1.7 in LR 2.2). FIFA pot was dropped from LR 2.2 because its VIF with rank was about 7.
- **Evaluation.** Hold-out split and 10-fold cross-validation, compared with a baseline that always predicts the training mean. In LR 2.2 each match appears twice as mirrored rows, so all splits are **grouped by match** (`GroupShuffleSplit`, `GroupKFold`).
- **Improvement.** Replacing the rank difference with the log of the rank ratio, `log(opponent rank / team rank)`. In LR 2.1, backward elimination was also applied.
- **Seeds.** `random_state=1` for the LR 2.1 split and folds; `42` for LR 2.2.

## Main results

All cross-validated figures below can be reproduced by running the notebooks.

| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| LR 2.1 baseline (predict mean) | 2.045 | 1.605 | −0.031 |
| LR 2.1 all 8 predictors | 1.762 | 1.397 | 0.235 |
| LR 2.1 with log rank ratio | 1.734 | 1.345 | 0.259 |
| **LR 2.1 final (log rank ratio only)** | **1.608** | **1.230** | **0.363** |
| LR 2.2 Model 1 (8 predictors) | 1.277 | 0.971 | 0.156 |
| **LR 2.2 Model 2 (log rank ratio)** | **1.261** | **0.935** | **0.177** |
| LR 2.2.3 extension (9 predictors) | 1.201 | 0.917 | 0.254 |

- The log rank ratio is the only predictor that stays significant in both LR 2.1 and LR 2.2.
- LR 2.1 final model: `goal difference = 0.137 + 0.802 × log(away rank / home rank)`; it picks the correct winner in 86% of the 80 non-drawn matches (cross-validated).
- The LR 2.2.3 extension is exploratory: its player-level feature is built from tournament-wide totals that include the match being predicted, so its gain is probably optimistic.

## Setup

Requires Python 3.12 or newer. Dependencies are declared in `pyproject.toml`: `jupyter`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `seaborn`, `statsmodels`.

```bash
# with uv
uv sync

# or with pip
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install jupyter matplotlib numpy pandas scikit-learn seaborn statsmodels
```

## Running the notebooks

Start Jupyter and open each notebook from its own folder, because the notebooks use relative paths.

```bash
jupyter notebook
```

- **LR 2.1** reads from `../data/` and runs as is.
- **LR 2.2** reads from `../processed/`, which is not a folder in this repository. Either create it from the data folder before running, or change the paths in the load cell and the save cell to `../data/`:
  ```bash
  mkdir processed && cp data/*.csv processed/
  ```
- **LR 2.2.3** reads its CSV files from the notebook's own folder. Copy them there first:
  ```bash
  cp data/{model_2_2,matches_clean,players_clean,team_matches_clean}.csv linear_regression_2.2/
  ```

## Limitations

- One tournament, 104 matches: small samples make hold-out estimates noisy.
- Goals are small whole numbers, so ordinary least squares is only an approximation (a count model such as Poisson regression would suit LR 2.2 better), and the two rows from each match in LR 2.2 are not independent.
- Backward elimination in LR 2.1 used p-values from all matches before cross-validation, so its cross-validated R² is probably somewhat optimistic.
- A linear model cannot predict a draw directly.