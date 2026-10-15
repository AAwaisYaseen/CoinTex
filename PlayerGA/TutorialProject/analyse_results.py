from pathlib import Path

# Install once: pip install pandas matplotlib
# Run with: python3 analyse_results.py

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "results" / "final_comparison_results.csv"
OUTPUT_DIR = BASE_DIR / "results" / "analysis"

NAMES = {
    "original_greedy": "Greedy",
    "hazard_aware_astar_optimized": "Hazard-Aware A*",
    "genetic_algorithm": "Genetic Algorithm",
}

COLOURS = {
    "Greedy": "#E69F00",
    "Hazard-Aware A*": "#0072B2",
    "Genetic Algorithm": "#009E73",
}


def save_table(data, title, filename, note=None):
    rows, columns = data.shape
    fig_width = max(9, columns * 1.6)
    fig_height = max(2.8, rows * 0.48 + 1.5)

    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    axis.axis("off")
    axis.set_title(title, fontsize=15, fontweight="bold", pad=16)

    table = axis.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if row == 0:
            cell.set_facecolor("#17365D")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6F8")

    if note:
        axis.text(0.5, 0.05, note, ha="center", fontsize=9, color="#444444")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_FILE)
    data["algorithm_name"] = data["algorithm"].map(NAMES)
    data["is_win"] = data["result"].eq("WIN")
    data["is_death"] = ~data["is_win"]

    expected_runs = data.groupby(["algorithm", "level"]).size()
    if len(data) != 240 or not expected_runs.eq(10).all():
        raise ValueError("Expected 10 runs for each algorithm and level.")

    wins = data[data["is_win"]]
    order = list(NAMES.values())
    levels = sorted(data["level"].unique())

    overall = data.groupby("algorithm_name").agg(
        runs=("result", "size"),
        wins=("is_win", "sum"),
        success_rate=("is_win", "mean"),
    )
    overall["mean_win_time"] = wins.groupby("algorithm_name")["time_seconds"].mean()
    overall = overall.reindex(order)

    success_level = (
        data.groupby(["level", "algorithm_name"])["is_win"]
        .mean()
        .mul(100)
        .unstack()
        .reindex(index=levels, columns=order)
    )
    time_level = (
        wins.groupby(["level", "algorithm_name"])["time_seconds"]
        .mean()
        .unstack()
        .reindex(index=levels, columns=order)
    )
    deaths_level = (
        data.groupby(["level", "algorithm_name"])["is_death"]
        .sum()
        .unstack()
        .reindex(index=levels, columns=order)
    )
    hardest_levels = [18, 24, 22]
    detected_hardest = deaths_level.sum(axis=1).sort_values(ascending=False).head(3).index.tolist()
    if set(detected_hardest) != set(hardest_levels):
        raise ValueError(f"Hardest levels changed: {detected_hardest}")

    collision = data.groupby("algorithm_name").agg(
        mean_total_hits=("total_hits", "mean"),
        mean_monster_hits=("monster_hits", "mean"),
        mean_fire_hits=("fire_hits", "mean"),
    ).reindex(order)

    overall_csv = overall.reset_index().rename(columns={
        "algorithm_name": "Algorithm",
        "runs": "Runs",
        "wins": "Wins",
        "success_rate": "Success Rate",
        "mean_win_time": "Mean Successful Time (s)",
    })
    overall_csv["Success Rate"] *= 100
    overall_csv = overall_csv[[
        "Algorithm",
        "Runs",
        "Wins",
        "Success Rate",
        "Mean Successful Time (s)",
    ]]
    overall_csv.to_csv(OUTPUT_DIR / "algorithm_summary.csv", index=False, float_format="%.2f")
    deaths_level.loc[hardest_levels].to_csv(OUTPUT_DIR / "hardest_level_deaths.csv", float_format="%.0f")
    collision.to_csv(OUTPUT_DIR / "collision_summary.csv", float_format="%.2f")

    overall_table = overall_csv.copy()
    overall_table["Success Rate"] = overall_table["Success Rate"].map(lambda value: f"{value:.1f}%")
    overall_table["Mean Successful Time (s)"] = overall_table["Mean Successful Time (s)"].map(lambda value: f"{value:.2f}")
    overall_table.columns = [
        "Algorithm",
        "Runs",
        "Wins",
        "Success Rate",
        "Mean Win Time (s)",
    ]
    save_table(
        overall_table,
        "Overall Algorithm Performance",
        "table_algorithm_summary.png",
        "Mean completion time is calculated from successful runs only.",
    )

    collision_table = collision.reset_index().rename(columns={
        "algorithm_name": "Algorithm",
        "mean_total_hits": "Average Total Hits",
        "mean_monster_hits": "Average Monster Hits",
        "mean_fire_hits": "Average Fire Hits",
    })
    for column in collision_table.columns[1:]:
        collision_table[column] = collision_table[column].map(lambda value: f"{value:.2f}")
    save_table(
        collision_table,
        "Collision Comparison",
        "table_collision_summary.png",
        "Averages include both successful and failed runs.",
    )

    x = np.arange(len(levels))
    width = 0.25

    fig, axis = plt.subplots(figsize=(11, 6))
    for index, algorithm in enumerate(order):
        bars = axis.bar(
            x + (index - 1) * width,
            success_level[algorithm],
            width,
            label=algorithm,
            color=COLOURS[algorithm],
        )
        labels = [f"{value:.0f}%" if value < 100 else "" for value in success_level[algorithm]]
        axis.bar_label(bars, labels=labels, padding=2, fontsize=7)
    axis.set_title("Success Rate by Level", fontweight="bold")
    axis.set_xlabel("Level")
    axis.set_ylabel("Success rate (%)")
    axis.set_xticks(x, levels)
    axis.set_ylim(0, 110)
    axis.legend(frameon=False, ncols=3, loc="upper center")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_success_rate_by_level.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6))
    for algorithm in order:
        axis.plot(
            levels,
            time_level[algorithm],
            marker="o",
            linewidth=2.2,
            label=algorithm,
            color=COLOURS[algorithm],
        )
    axis.set_title("Mean Completion Time by Level\nSuccessful runs only", fontweight="bold")
    axis.set_xlabel("Level")
    axis.set_ylabel("Time (seconds)")
    axis.set_xticks(levels)
    axis.legend(frameon=False, ncols=3, loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_completion_time_by_level.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    hard_x = np.arange(len(hardest_levels))
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for index, algorithm in enumerate(order):
        bars = axis.bar(
            hard_x + (index - 1) * width,
            deaths_level.loc[hardest_levels, algorithm],
            width,
            label=algorithm,
            color=COLOURS[algorithm],
        )
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    axis.set_title("Deaths on the Three Hardest Levels", fontweight="bold")
    axis.set_xlabel("Level")
    axis.set_ylabel("Deaths from 10 runs")
    axis.set_xticks(hard_x, hardest_levels)
    axis.set_ylim(0, 10)
    axis.legend(frameon=False, ncols=3, loc="upper center")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_deaths_hardest_levels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    old_outputs = [
        "chart_collisions_by_level.png",
        "chart_level_difficulty_and_astar.png",
        "chart_successful_completion_time.png",
        "level_difficulty_summary.csv",
        "table_level_difficulty.png",
    ]
    for filename in old_outputs:
        (OUTPUT_DIR / filename).unlink(missing_ok=True)

    print("Analysis completed.")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("Deaths on the hardest levels:")
    print(deaths_level.loc[hardest_levels].to_string())


if __name__ == "__main__":
    main()
