# COMP6009 Assignment A - result analysis
#
# Install once: pip install pandas matplotlib
# Run with:     python3 analyse_results.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FOLDER = Path(__file__).resolve().parent
FINAL_FILE = FOLDER / "results" / "final_comparison_results.csv"
PILOT_FILE = FOLDER / "results" / "astar_optimisation_pilot.csv"
OUTPUT_FOLDER = FOLDER / "results" / "analysis"

NAMES = {
    "original_greedy": "Greedy",
    "hazard_aware_astar_optimized": "Hazard-Aware A*",
    "genetic_algorithm": "Genetic Algorithm",
}
ORDER = ["Greedy", "Hazard-Aware A*", "Genetic Algorithm"]
COLOURS = {
    "Greedy": "#E69F00",
    "Hazard-Aware A*": "#0072B2",
    "Genetic Algorithm": "#009E73",
}

# chart style used for every chart
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#BBBBBB",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#E5E5E5",
    "axes.axisbelow": True,
    "ytick.color": "#666666",
})


def load_results(file_path):
    data = pd.read_csv(file_path)
    data["is_win"] = data["result"] == "WIN"
    data["coin_completion"] = 100 * data["coins_collected"] / data["total_coins"]
    return data


def make_summary(data, group_by):
    summary = data.groupby(group_by).agg(
        runs=("result", "size"),
        wins=("is_win", "sum"),
        success_rate=("is_win", "mean"),
        avg_hits=("total_hits", "mean"),
        avg_monster_hits=("monster_hits", "mean"),
        avg_fire_hits=("fire_hits", "mean"),
        avg_life=("remaining_life_percent", "mean"),
        avg_coin_completion=("coin_completion", "mean"),
    )
    summary["success_rate"] = summary["success_rate"] * 100

    # time is only averaged for winning runs
    winning_runs = data[data["is_win"]]
    summary["mean_win_time"] = winning_runs.groupby(group_by)["time_seconds"].mean()
    return summary


def add_titles(fig, title, subtitle):
    height = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.1 / height, ha="left", va="top",
                 fontsize=15, fontweight="bold")
    fig.text(0.01, 1 - 0.42 / height, subtitle, ha="left", va="top",
             fontsize=11, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / height))


def overall_chart(overall):
    # success, time and hits
    panels = [
        ("success_rate", "Success rate (%)\nhigher is better", "{:g}%"),
        ("mean_win_time", "Time to collect all coins (s)\nlower is better", "{:.2f} s"),
        ("avg_hits", "Hits per run\nlower is better", "{:.2f}"),
    ]
    tick_names = ["Greedy", "Hazard-Aware\nA*", "Genetic\nAlgorithm"]
    colours = [COLOURS[name] for name in ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    for axis, (column, label, number_format) in zip(axes, panels):
        values = overall.loc[ORDER, column]
        bars = axis.bar(range(3), values, width=0.6, color=colours)

        bar_labels = []
        for value in values:
            bar_labels.append(number_format.format(value))
        axis.bar_label(bars, labels=bar_labels, padding=4, fontweight="bold")

        axis.set_title(label, fontsize=12, loc="left")
        axis.set_xticks(range(3), tick_names)
        if column == "success_rate":
            axis.set_ylim(0, 115)
        else:
            axis.set_ylim(0, values.max() * 1.22)

    add_titles(
        fig,
        "Hazard-Aware A* is the most reliable and safest; Greedy is fastest when it wins",
        "80 runs per algorithm (8 levels x 10 runs). Time uses winning runs only.",
    )
    fig.savefig(OUTPUT_FOLDER / "chart_overall_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def level_chart(table, labels, title, subtitle, y_label, file_name, y_max):
    levels = list(table.index)
    x = np.arange(len(levels))
    width = 0.27

    fig, axis = plt.subplots(figsize=(13, 5.8))
    for i, name in enumerate(ORDER):
        bars = axis.bar(x + (i - 1) * width, table[name], width * 0.92,
                        label=name, color=COLOURS[name])
        axis.bar_label(bars, labels=labels[name], padding=3, fontsize=8.5)

    level_names = []
    for level in levels:
        level_names.append(f"Level {level}")
    axis.set_xticks(x, level_names)
    axis.set_ylabel(y_label, color="#666666")
    axis.set_ylim(0, y_max)
    axis.legend(frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0, 1.02))

    add_titles(fig, title, subtitle)
    fig.savefig(OUTPUT_FOLDER / file_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Greedy vs Hazard-Aware A* vs GA
    final = load_results(FINAL_FILE)
    final["algorithm_name"] = final["algorithm"].map(NAMES)

    # Check the dataset is complete: 3 algorithms x 8 levels x 10 runs
    runs_per_level = final.groupby(["algorithm", "level"]).size()
    if len(final) != 240 or not (runs_per_level == 10).all():
        raise ValueError("Expected 10 runs for each algorithm and level.")

    overall = make_summary(final, "algorithm_name").loc[ORDER]
    by_level = make_summary(final, ["level", "algorithm_name"])
    overall.round(2).to_csv(OUTPUT_FOLDER / "algorithm_summary.csv")
    by_level.round(2).to_csv(OUTPUT_FOLDER / "level_summary.csv")

    overall_chart(overall)

    success = by_level["success_rate"].unstack()[ORDER]
    success_labels = {}
    for name in ORDER:
        success_labels[name] = [f"{value:.0f}%" for value in success[name]]
    level_chart(
        success, success_labels,
        "Success rate by level",
        "A* leads on the hard Levels 18 and 24; GA edges ahead on Levels 4 and 10, "
        "Greedy on Level 20. 10 runs per bar.",
        "Success rate (%)", "chart_success_rate_by_level.png", 118,
    )

    # completion time by level (winning runs only)
    times = by_level["mean_win_time"].unstack()[ORDER]
    wins = by_level["wins"].unstack()[ORDER]
    time_labels = {}
    for name in ORDER:
        time_labels[name] = []
        for time_value, win_count in zip(times[name], wins[name]):
            label = f"{time_value:.1f}"
            if win_count <= 3:
                label += f"\nn={int(win_count)}"  # show when the average uses few wins
            time_labels[name].append(label)
    level_chart(
        times, time_labels,
        "Time to collect all coins by level",
        "Mean seconds on winning runs only. n = number of wins averaged, "
        "shown where it is 3 or fewer.",
        "Seconds", "chart_completion_time_by_level.png", times.max().max() * 1.3,
    )
 
    hits = by_level["avg_hits"].unstack()[ORDER]
    hits_labels = {}
    for name in ORDER:
        hits_labels[name] = [f"{value:.1f}" for value in hits[name]]
    level_chart(
        hits, hits_labels,
        "Hits per run by level",
        "A* has the fewest hits on every level, even where it won fewer runs. "
        "Average of 10 runs per bar (wins and losses).",
        "Average hits per run", "chart_hits_by_level.png", hits.max().max() * 1.25,
    )

    # Initial vs optimised Hazard-Aware A*
    pilot = load_results(PILOT_FILE)
    pilot["version"] = pilot["algorithm"].map({
        "hazard_aware_astar": "Initial A*",
        "hazard_aware_astar_optimized": "Optimised A*",
    })
    if not (pilot.groupby("version").size() == 25).all():
        raise ValueError("Expected 25 pilot runs for each A* version.")

    pilot_overall = make_summary(pilot, "version")
    pilot_by_level = make_summary(pilot, ["level", "version"])
    pilot_overall.round(2).to_csv(OUTPUT_FOLDER / "astar_optimisation_summary.csv")
    pilot_by_level.round(2).to_csv(OUTPUT_FOLDER / "astar_optimisation_by_level.csv")

    columns = ["runs", "wins", "success_rate", "mean_win_time", "avg_hits"]
    print("Analysis completed. Files saved in:", OUTPUT_FOLDER)
    print(overall[columns].round(2).to_string())
    print(pilot_overall[columns].round(2).to_string())


if __name__ == "__main__":
    main()
