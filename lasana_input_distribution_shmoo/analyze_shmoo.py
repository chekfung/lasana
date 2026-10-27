"""Summarize and plot one completed LASANA input-distribution shmoo."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Select the completed shmoo and analysis settings

parser = argparse.ArgumentParser(description="Analyze a completed LASANA input-distribution shmoo.")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--runs-per-point", type=int, default=100)
parser.add_argument("--timesteps-per-run", type=int, default=100)
parser.add_argument("--high-membrane-threshold", type=float, default=0.5)
parser.add_argument("--colormap", default="Blues")
arguments = parser.parse_args()

output_directory = Path(arguments.output_dir).expanduser().resolve()
manifest = pd.read_csv(output_directory / "shmoo_points.csv")
color_map = plt.get_cmap(arguments.colormap)

if len(manifest) != 19:
    raise RuntimeError(f"Expected 18 Poisson points and one legacy point, found {len(manifest)}")


# -----------------------------
# Read and summarize every shmoo point

summaries = []
plot_data = {}

for point in manifest.to_dict("records"):
    run_directory = Path(point["run_directory"])

    # Load the generated inputs, run totals, and parsed circuit events.
    input_tracking = pd.read_csv(run_directory / "input_tracking.csv")
    randomized_log = pd.read_csv(run_directory / "randomized_output_log.csv")
    events = pd.read_csv(run_directory / "spiking_neuron_dataset.csv")

    # Each point must contain exactly the configured number of completed runs.
    expected_runs = set(range(arguments.runs_per_point))

    if set(randomized_log["Run_Number"]) != expected_runs:
        raise RuntimeError(
            f"{point['run_name']} does not contain {arguments.runs_per_point} logged runs"
        )

    if set(events["Run_Number"]) != expected_runs:
        raise RuntimeError(
            f"{point['run_name']} does not contain {arguments.runs_per_point} parsed runs"
        )

    # Confirm that the timestep firing counts agree with the total for each run.
    tracked_firings = input_tracking.groupby("Run_Number")["Num_Firings"].sum()
    tracked_firings = tracked_firings.reindex(
        range(arguments.runs_per_point), fill_value=0,
    ).astype(int)
    logged_firings = randomized_log.set_index("Run_Number")["Num_Input_Spikes_I_inj"].astype(int)

    if not tracked_firings.equals(logged_firings):
        raise RuntimeError(f"{point['run_name']} has inconsistent firing totals")

    # input_tracking contains occupied timesteps only. Add the omitted zeros so
    # the firing-count histogram represents every digital timestep.
    total_timesteps = arguments.runs_per_point * arguments.timesteps_per_run
    counts = np.concatenate([
        input_tracking["Num_Firings"].to_numpy(dtype=int),
        np.zeros(total_timesteps - len(input_tracking), dtype=int),
    ])

    # Separate input events by whether they caused an output spike.
    input_events = events[events["Event_Type"].isin(["in-no_out", "in-out"])]
    in_out = input_events[input_events["Event_Type"] == "in-out"]
    in_no_out = input_events[input_events["Event_Type"] == "in-no_out"]

    # A high state means that the membrane was already above the configured
    # threshold immediately before this timestep's aggregate input was applied.
    high_in_out = (
        in_out["Cap_Voltage_At_Input_Start"] > arguments.high_membrane_threshold
    )
    high_in_no_out = (
        in_no_out["Cap_Voltage_At_Input_Start"] > arguments.high_membrane_threshold
    )

    # Each high-membrane percentage uses its own event class as the denominator.
    summaries.append({
        "run_name": point["run_name"],
        "input_generation": point["input_generation"],
        "weight_family": point["weight_family"],
        "poisson_lambda": float(point["poisson_lambda"]),
        "weight_scale_name": point["weight_scale_name"],
        "weight_scale": float(point["weight_scale"]),
        "zero_timestep_percent": 100 * np.mean(counts == 0),
        "active_timesteps": len(input_tracking),
        "total_input_firings": int(input_tracking["Num_Firings"].sum()),
        "in_out_events": len(in_out),
        "in_no_out_events": len(in_no_out),
        "output_spike_percent": 100 * len(in_out) / len(input_events),
        "high_membrane_in_out_events": int(high_in_out.sum()),
        "high_membrane_in_out_percent": 100 * high_in_out.mean(),
        "high_membrane_in_no_out_events": int(high_in_no_out.sum()),
        "high_membrane_in_no_out_percent": 100 * high_in_no_out.mean(),
    })

    # Preserve only the raw arrays used by the histograms below.
    plot_data[point["run_name"]] = {
        "counts": counts,
        "weights": input_tracking["Value"].to_numpy(),
        "in_out_membrane_start": in_out["Cap_Voltage_At_Input_Start"].to_numpy(),
        "in_no_out_membrane_start": in_no_out["Cap_Voltage_At_Input_Start"].to_numpy(),
    }


# -----------------------------
# Save the numerical summary

summary = pd.DataFrame(summaries).sort_values(
    ["input_generation", "weight_family", "poisson_lambda", "weight_scale"],
)

poisson_summary = summary[summary["input_generation"] == "poisson"]
legacy_summary = summary[summary["input_generation"] == "legacy"].iloc[0]

tables_directory = output_directory / "tables"
plots_directory = output_directory / "plots"
tables_directory.mkdir(exist_ok=True)
plots_directory.mkdir(exist_ok=True)

summary.to_csv(tables_directory / "shmoo_summary.csv", index=False)


# -----------------------------
# Plot the simultaneous firing-count distributions

# Count sampling depends only on lambda, not on the weight distribution. Plot
# one representative point per lambda beside the original legacy generator.
lambdas = sorted(poisson_summary["poisson_lambda"].unique())
figure, axes = plt.subplots(1, len(lambdas) + 1, figsize=(18, 4), squeeze=False)

legacy_counts = plot_data[legacy_summary["run_name"]]["counts"]
axes[0, 0].hist(
    legacy_counts,
    bins=np.arange(legacy_counts.max() + 2) - 0.5,
    density=True,
    color=color_map(0.65),
    edgecolor="white",
)
axes[0, 0].set_title("Legacy spike-map generator")
axes[0, 0].set_xlabel("Simultaneous input firings in one timestep")
axes[0, 0].set_ylabel("Fraction of timesteps")

for axis, poisson_lambda in zip(axes[0, 1:], lambdas):
    run_name = poisson_summary.loc[
        poisson_summary["poisson_lambda"] == poisson_lambda, "run_name"
    ].iloc[0]
    counts = plot_data[run_name]["counts"]

    axis.hist(
        counts,
        bins=np.arange(counts.max() + 2) - 0.5,
        density=True,
        color=color_map(0.65),
        edgecolor="white",
    )
    axis.set_title(f"Poisson λ = {poisson_lambda:g}")
    axis.set_xlabel("Simultaneous input firings in one timestep")
    axis.set_ylabel("Fraction of timesteps")

figure.suptitle("Realized Num_Firings distributions (all use DELAY_RATIO = 0.3)")
figure.tight_layout()
figure.savefig(plots_directory / "num_firings_histograms.png", dpi=200)
plt.close(figure)


# -----------------------------
# Plot the aggregate weight distributions

# Each figure holds one lambda. Rows select the weight family, while columns
# select the normal sigma or uniform half-width.
for poisson_lambda in sorted(poisson_summary["poisson_lambda"].unique()):
    lambda_rows = poisson_summary[poisson_summary["poisson_lambda"] == poisson_lambda]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row_index, family in enumerate(["normal", "uniform"]):
        rows = lambda_rows[lambda_rows["weight_family"] == family].sort_values("weight_scale")

        for column_index, (_, row) in enumerate(rows.iterrows()):
            axis = axes[row_index, column_index]

            # Value is the aggregate weight applied during one occupied timestep.
            # It is not an individual presynaptic weight.
            weights = plot_data[row["run_name"]]["weights"]
            symbol = "σ" if family == "normal" else "a"

            axis.hist(weights, bins=50, density=True, color=color_map(0.65), edgecolor="none")
            axis.set_title(f"{family.title()} {symbol}={row['weight_scale']:g}")
            axis.set_xlabel("Aggregate weight applied in one occupied timestep")
            axis.set_ylabel("Density")

    figure.suptitle(f"Applied aggregate input weights — Poisson λ={poisson_lambda:g}")
    figure.text(
        0.5,
        0.01,
        "Normal: sum N independent N(0, σ²) weights.  Uniform: one aggregate draw U(-aN, aN).",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))

    label = str(poisson_lambda).replace(".", "p")
    figure.savefig(plots_directory / f"aggregate_weight_histograms_lambda_{label}.png", dpi=200)
    plt.close(figure)


# -----------------------------
# Plot the pre-input membrane-state distributions

# Overlay in-out and in-no_out events so their state coverage can be compared
# without mixing their percentage denominators.
for poisson_lambda in sorted(poisson_summary["poisson_lambda"].unique()):
    lambda_rows = poisson_summary[poisson_summary["poisson_lambda"] == poisson_lambda]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row_index, family in enumerate(["normal", "uniform"]):
        rows = lambda_rows[lambda_rows["weight_family"] == family].sort_values("weight_scale")

        for column_index, (_, row) in enumerate(rows.iterrows()):
            axis = axes[row_index, column_index]
            point = plot_data[row["run_name"]]
            in_out = point["in_out_membrane_start"]
            in_no_out = point["in_no_out_membrane_start"]

            # Both event classes use identical bin boundaries in this panel.
            values = np.concatenate([in_out, in_no_out])
            bins = np.linspace(values.min(), values.max(), 50)

            # Density normalizes each class separately. The legend also reports
            # the raw count so a small event class cannot be mistaken for broad coverage.
            axis.hist(
                in_no_out,
                bins=bins,
                density=True,
                alpha=0.60,
                color=color_map(0.45),
                label=(
                    f"in-no_out: n={len(in_no_out)}, "
                    f"high={row['high_membrane_in_no_out_percent']:.1f}%"
                ),
            )
            axis.hist(
                in_out,
                bins=bins,
                density=True,
                alpha=0.60,
                color=color_map(0.80),
                label=(
                    f"in-out: n={len(in_out)}, "
                    f"high={row['high_membrane_in_out_percent']:.1f}%"
                ),
            )
            axis.axvline(
                arguments.high_membrane_threshold,
                color="black",
                linestyle="--",
                label=f"high membrane > {arguments.high_membrane_threshold:g} V",
            )

            symbol = "σ" if family == "normal" else "a"
            axis.set_title(f"{family.title()} {symbol}={row['weight_scale']:g}")
            axis.set_xlabel("Membrane voltage immediately before input (V)")
            axis.set_ylabel("Density within event class")
            axis.legend(fontsize=7)

    figure.suptitle(f"Pre-input membrane-state coverage — Poisson λ={poisson_lambda:g}")
    figure.text(
        0.5,
        0.01,
        "Membrane start = Cap_Voltage_At_Input_Start. Percentages use each event class as its denominator.",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))

    label = str(poisson_lambda).replace(".", "p")
    figure.savefig(plots_directory / f"pre_input_membrane_histograms_lambda_{label}.png", dpi=200)
    plt.close(figure)


# -----------------------------
# Plot high-membrane coverage across the parameter grid

# Show raw counts beside within-class percentages. A large percentage based on
# very few events does not provide the same training coverage as many rows.
coverage_metrics = [
    ("high_membrane_in_out_events", "High-membrane in-out count", ".0f"),
    ("high_membrane_in_out_percent", "High-membrane share of in-out (%)", ".1f"),
    ("high_membrane_in_no_out_events", "High-membrane in-no_out count", ".0f"),
    ("high_membrane_in_no_out_percent", "High-membrane share of in-no_out (%)", ".1f"),
]

figure, axes = plt.subplots(2, 4, figsize=(19, 8))

for row_index, family in enumerate(["normal", "uniform"]):
    family_rows = poisson_summary[poisson_summary["weight_family"] == family]

    for column_index, (metric, title, number_format) in enumerate(coverage_metrics):
        axis = axes[row_index, column_index]

        # Turn the 18 Poisson points into a lambda-by-weight grid.
        values = family_rows.pivot(
            index="poisson_lambda",
            columns="weight_scale",
            values=metric,
        ).sort_index()

        image = axis.imshow(values.to_numpy(), aspect="auto", origin="lower", cmap=color_map)

        axis.set_xticks(range(len(values.columns)), [f"{value:g}" for value in values.columns])
        axis.set_yticks(range(len(values.index)), [f"{value:g}" for value in values.index])
        axis.set_xlabel("Normal σ" if family == "normal" else "Uniform half-width a")
        axis.set_ylabel("Poisson λ")
        axis.set_title(f"{family.title()}: {title}")

        # Print every value and switch text color only to retain contrast.
        text_color_threshold = np.nanmax(values.to_numpy()) * 0.55

        for y in range(len(values.index)):
            for x in range(len(values.columns)):
                value = values.iloc[y, x]
                text_color = "white" if value > text_color_threshold else "black"
                axis.text(
                    x,
                    y,
                    format(value, number_format),
                    ha="center",
                    va="center",
                    color=text_color,
                )

        figure.colorbar(image, ax=axis)

figure.suptitle(
    "Coverage of high pre-input membrane voltage "
    f"(> {arguments.high_membrane_threshold:g} V)"
)
figure.text(
    0.5,
    0.01,
    "in-out percentages divide by all in-out events; in-no_out percentages divide by all in-no_out events.",
    ha="center",
)
figure.tight_layout(rect=(0, 0.04, 1, 0.96))
figure.savefig(plots_directory / "high_pre_input_membrane_coverage.png", dpi=200)
plt.close(figure)


# -----------------------------
# Compare the original generator with matched uniform Poisson points

# Hold the aggregate-uniform sampler at [-2N, 2N]. This leaves the spike-map
# generator as the intended comparison: legacy versus the three Poisson lambdas.
comparison = pd.concat([
    summary[summary["input_generation"] == "legacy"],
    poisson_summary[
        (poisson_summary["weight_family"] == "uniform")
        & (poisson_summary["weight_scale"] == 2.0)
    ].sort_values("poisson_lambda"),
])
comparison_labels = ["Legacy"] + [
    f"Poisson λ={value:g}" for value in comparison["poisson_lambda"].iloc[1:]
]

comparison_metrics = [
    ("high_membrane_in_out_events", "High-membrane in-out count"),
    ("high_membrane_in_out_percent", "High-membrane share of in-out (%)"),
    ("high_membrane_in_no_out_events", "High-membrane in-no_out count"),
    ("high_membrane_in_no_out_percent", "High-membrane share of in-no_out (%)"),
]

figure, axes = plt.subplots(2, 2, figsize=(13, 9))

for axis, (metric, title) in zip(axes.flat, comparison_metrics):
    bars = axis.bar(
        comparison_labels,
        comparison[metric],
        color=[color_map(value) for value in np.linspace(0.35, 0.80, len(comparison))],
    )
    axis.bar_label(bars, fmt="%.1f" if "percent" in metric else "%.0f")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)

figure.suptitle("Legacy versus Poisson coverage with aggregate-uniform [-2N, 2N] weights")
figure.text(
    0.5,
    0.01,
    "All points use 100 runs, fixed knobs, and DELAY_RATIO = 0.3.",
    ha="center",
)
figure.tight_layout(rect=(0, 0.04, 1, 0.96))
figure.savefig(plots_directory / "legacy_uniform_comparison.png", dpi=200)
plt.close(figure)


print(f"Completed analysis: {output_directory}")
