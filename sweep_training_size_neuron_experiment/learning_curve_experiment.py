#!/usr/bin/env python3
"""Plot model error as the spiking-neuron training set grows."""

import argparse
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.interpolate import NearestNDInterpolator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


# Locate the repository and import the active spiking-neuron configuration
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import config_spiking_neuron as config


# ----------------------------- BEGIN Experiment Settings ------------------------------ #
TRAIN_SIZES = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 14000]    # Number of circuit runs used for training
SEEDS = [51, 52, 53, 54, 55]                                         # Repeated runs used to measure variation
ZOOMED_PLOT_MAX_TRAIN_RUNS = 3000                                     # Upper limit for the additional zoomed learning-curve figure

# Prediction tasks included in the learning-curve experiment
TASKS = [
    "latency",
    "dynamic_energy",
    "static_energy",
    "neuron_state",
    "spike_behavior",
]

# Model types compared at every training size
METHODS = ["baseline", "nearest", "linear", "catboost", "mlp"]

# Labels for the plotting behavior
TASK_LABELS = {
    "dynamic_energy": "Dynamic Energy",
    "latency": "Latency",
    "neuron_state": "Neuron State",
    "spike_behavior": "Spike / No Spike",
    "static_energy": "Static Energy",
}
METHOD_LABELS = {
    "baseline": "Mean/Majority Baseline",
    "nearest": "NN Interpolation",
    "linear": "OLS/Logistic",
    "catboost": "CatBoost",
    "mlp": "MLP",
}

OUTPUT_DIR = os.path.join(ROOT, "tmp", "learning_curves")
RAW_RESULTS = os.path.join(OUTPUT_DIR, "learning_curve_raw.csv")
SUMMARY_RESULTS = os.path.join(OUTPUT_DIR, "learning_curve_summary.csv")
FIGURE = os.path.join(OUTPUT_DIR, "learning_curves.png")
ZOOMED_FIGURE = os.path.join(OUTPUT_DIR, "learning_curves_0_to_3000.png")
CONFIG_SNAPSHOT = os.path.join(OUTPUT_DIR, "config_spiking_neuron.py")
# ----------------------------- END Experiment Settings ------------------------------ #


def load_data():
    # Load only the columns used by the five prediction tasks
    columns = set(config.LIST_OF_COLUMNS_X) | {
        "Run_Number",
        "Event_Type",
        "Energy",
        "Latency",
        "Cap_Voltage_At_Output_End",
    }
    path = os.path.join(ROOT, "data", config.RUN_NAME, config.DF_FILENAME)
    data = pd.read_csv(path, usecols=sorted(columns))
    data["Latency"] *= 1e9       # Convert latency from seconds to ns
    data["Energy"] *= 1e12       # Convert energy from joules to pJ
    return data


def split_runs(data, seed):
    # Match the seeded run-level splitting used by the existing model scripts
    runs = np.sort(data["Run_Number"].unique())
    training_and_validation_runs, test_runs = train_test_split(
        runs,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=seed,
    )
    adjusted_validation_size = config.VALIDATION_SPLIT / (1 - config.TRAIN_TEST_SPLIT)
    training_runs, validation_runs = train_test_split(
        training_and_validation_runs,
        test_size=adjusted_validation_size,
        random_state=seed,
    )

    # Randomly permute training runs based on seed so that different seeds get different sets
    # Larger training sets are additive, meaning they contain the prior smaller dataset + more data
    training_runs = np.random.RandomState(seed).permutation(np.sort(training_runs))
    return training_runs, validation_runs, test_runs


def prepare_tasks(data):
    # Convert event labels into the binary target used by the spike classifier
    spike_data = data[data["Event_Type"].isin(["in-out", "in-no_out"])].copy()
    spike_data["Spike"] = (spike_data["Event_Type"] == "in-out").astype(int)

    # Each tuple contains its filtered rows, target column, and classification flag
    return {
        "dynamic_energy": (data[data["Event_Type"] == "in-out"], "Energy", False),
        "latency": (data[data["Event_Type"] == "in-out"], "Latency", False),
        "neuron_state": (data, "Cap_Voltage_At_Output_End", False),
        "spike_behavior": (spike_data, "Spike", True),
        "static_energy": (
            data[data["Event_Type"].isin(["leak", "leak-2", "in-no_out"])],
            "Energy",
            False,
        ),
    }


def task_arrays(task_data, target, train_runs, validation_runs, test_runs):
    # Apply the shared run split after filtering the rows for one prediction task
    features = config.LIST_OF_COLUMNS_X
    train = task_data[task_data["Run_Number"].isin(train_runs)]
    validation = task_data[task_data["Run_Number"].isin(validation_runs)]
    test = task_data[task_data["Run_Number"].isin(test_runs)]

    return (
        train[features],
        train[target].to_numpy(),
        validation[features],
        validation[target].to_numpy(),
        test[features],
        test[target].to_numpy(),
    )


def predict(method, classification, arrays, scaler, seed):
    x_train, y_train, x_validation, y_validation, x_test, _ = arrays

    # Mean regression baseline or majority-class classification baseline
    if method == "baseline":
        training_start = time.time()
        if classification:
            values, counts = np.unique(y_train, return_counts=True)
            value = values[np.argmax(counts)]
        else:
            value = y_train.mean()
        training_time = time.time() - training_start

        inference_start = time.time()
        prediction = np.full(len(x_test), value)
        inference_time = time.time() - inference_start
        return prediction, training_time, inference_time

    # Table-based nearest-neighbor interpolation
    if method == "nearest":
        training_start = time.time()
        model = NearestNDInterpolator(x_train.to_numpy(), y_train)
        training_time = time.time() - training_start

        inference_start = time.time()
        prediction = model(x_test.to_numpy())
        inference_time = time.time() - inference_start
        return prediction, training_time, inference_time

    # Ordinary least squares for regression or logistic regression for classification
    if method == "linear":
        model = LogisticRegression() if classification else LinearRegression()

        training_start = time.time()
        model.fit(scaler.transform(x_train), y_train)
        training_time = time.time() - training_start

        inference_start = time.time()
        prediction = model.predict(scaler.transform(x_test))
        inference_time = time.time() - inference_start
        return prediction, training_time, inference_time

    # CatBoost uses the fixed validation partition for early stopping
    if method == "catboost":
        common = {
            "learning_rate": 0.1,
            "depth": 10,
            "l2_leaf_reg": 5,
            "subsample": 0.5,
            "verbose": False,
            "random_seed": seed,
            "allow_writing_files": False,
        }
        if classification:
            model = CatBoostClassifier(iterations=300, eval_metric="Logloss", **common)
        else:
            model = CatBoostRegressor(iterations=500, eval_metric="RMSE", **common)

        training_start = time.time()
        model.fit(
            x_train,
            y_train,
            eval_set=(x_validation, y_validation),
            early_stopping_rounds=50,
        )
        training_time = time.time() - training_start

        inference_start = time.time()
        prediction = model.predict(x_test)
        inference_time = time.time() - inference_start
        return prediction, training_time, inference_time

    # MLP keeps the requested training runs separate from the fixed validation set
    mlp_options = {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate_init": 0.01,
        "tol": 1e-5,
        "early_stopping": True,
        "validation_fraction": config.VALIDATION_SPLIT,
        "random_state": seed,
    }
    model = MLPClassifier(**mlp_options) if classification else MLPRegressor(**mlp_options)

    training_start = time.time()
    model.fit(scaler.transform(x_train), y_train)
    training_time = time.time() - training_start

    inference_start = time.time()
    prediction = model.predict(scaler.transform(x_test))
    inference_time = time.time() - inference_start
    return prediction, training_time, inference_time


def calculate_metrics(y_true, y_predicted, classification):
    # MSE and normalized MSE are shared by regression and classification tasks
    mse = np.mean((np.asarray(y_true) - np.asarray(y_predicted)) ** 2)
    variance = np.var(y_true)
    nmse = mse / variance

    metrics = {
        "mse": mse,
        "target_variance": variance,
        "nmse": nmse,
        "mae": np.nan,
        "mape_percent": np.nan,
        "r_squared": np.nan,
        "accuracy": np.nan,
        "roc_auc": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
    }

    if classification:
        metrics.update(
            accuracy=accuracy_score(y_true, y_predicted),
            roc_auc=roc_auc_score(y_true, y_predicted),
            precision=precision_score(y_true, y_predicted, zero_division=0),
            recall=recall_score(y_true, y_predicted, zero_division=0),
            f1=f1_score(y_true, y_predicted, zero_division=0),
        )
    else:
        metrics.update(
            mae=mean_absolute_error(y_true, y_predicted),
            mape_percent=mean_absolute_percentage_error(y_true, y_predicted) * 100,
            r_squared=r2_score(y_true, y_predicted),
        )

    return metrics


def plot_results(results):
    # Combine the five seeded runs into a mean line and standard-deviation band
    summary = (
        results.groupby(["method", "train_runs", "task"], as_index=False)
        .agg(
            mean_nmse=("nmse", "mean"),
            std_nmse=("nmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_mape_percent=("mape_percent", "mean"),
            std_mape_percent=("mape_percent", "std"),
            mean_r_squared=("r_squared", "mean"),
            std_r_squared=("r_squared", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_training_samples=("training_samples", "mean"),
            mean_validation_samples=("validation_samples", "mean"),
            mean_test_samples=("test_samples", "mean"),
            mean_training_time_seconds=("training_time_seconds", "mean"),
            std_training_time_seconds=("training_time_seconds", "std"),
            mean_inference_time_seconds=("inference_time_seconds", "mean"),
            std_inference_time_seconds=("inference_time_seconds", "std"),
            mean_total_time_seconds=("total_time_seconds", "mean"),
            std_total_time_seconds=("total_time_seconds", "std"),
            completed_seeds=("seed", "nunique"),
        )
        .sort_values(["method", "train_runs", "task"])
    )
    standard_deviation_columns = [column for column in summary.columns if column.startswith("std_")]
    summary[standard_deviation_columns] = summary[standard_deviation_columns].fillna(0)
    summary.to_csv(SUMMARY_RESULTS, index=False)

    # Use five NMSE panels and a separate accuracy panel for spike classification
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    (
        latency_axis,
        dynamic_energy_axis,
        static_energy_axis,
        neuron_state_axis,
        spike_nmse_axis,
        spike_accuracy_axis,
    ) = axes.ravel()

    # Each panel identifies its task, plotted metric, and standard-deviation columns
    panels = [
        (latency_axis, "latency", "mean_nmse", "std_nmse", "Latency", "Normalized MSE (MSE / test variance)"),
        (dynamic_energy_axis, "dynamic_energy", "mean_nmse", "std_nmse", "Dynamic Energy", "Normalized MSE (MSE / test variance)"),
        (static_energy_axis, "static_energy", "mean_nmse", "std_nmse", "Static Energy", "Normalized MSE (MSE / test variance)"),
        (neuron_state_axis, "neuron_state", "mean_nmse", "std_nmse", "Neuron State", "Normalized MSE (MSE / test variance)"),
        (spike_nmse_axis, "spike_behavior", "mean_nmse", "std_nmse", "Spike / No Spike — NMSE", "Normalized MSE (MSE / test variance)"),
        (spike_accuracy_axis, "spike_behavior", "mean_accuracy", "std_accuracy", "Spike / No Spike — Accuracy", "Accuracy"),
    ]

    # Give every method the same color in all six panels
    colors = dict(zip(METHODS, plt.get_cmap("tab10").colors))

    for axis, task, mean_column, std_column, title, y_label in panels:
        for method in METHODS:
            values = summary[(summary["task"] == task) & (summary["method"] == method)]

            # Plot against the actual number of training circuit runs
            x = values["train_runs"].to_numpy()
            mean = values[mean_column].to_numpy()
            std = values[std_column].to_numpy()

            # Each line is the seed mean and each shaded region is one standard deviation
            axis.plot(
                x,
                mean,
                marker="o",
                linewidth=2,
                color=colors[method],
                label=METHOD_LABELS[method],
            )

            lower_bound = mean - std
            upper_bound = mean + std
            if mean_column == "mean_accuracy":
                lower_bound = np.maximum(lower_bound, 0)
                upper_bound = np.minimum(upper_bound, 1)

            axis.fill_between(
                x,
                lower_bound,
                upper_bound,
                alpha=0.18,
                color=colors[method],
            )

        axis.set_title(title)
        axis.set_xlabel("Training circuit runs")
        axis.set_ylabel(y_label)
        axis.set_xlim(0, max(TRAIN_SIZES))
        axis.xaxis.set_major_locator(
            MaxNLocator(nbins=7, integer=True, steps=[1, 2, 5, 10])
        )
        axis.tick_params(axis="x", labelbottom=True)
        axis.tick_params(axis="y", labelleft=True)
        axis.grid(True, alpha=0.25)

    # Accuracy is a proportion, so keep its scale fixed from zero to one
    spike_accuracy_axis.set_ylim(0, 1)

    # Use one legend for the whole figure so all six panels remain available for data
    figure.legend(
        *latency_axis.get_legend_handles_labels(),
        loc="lower center",
        ncol=len(METHODS),
        frameon=False,
    )

    # Write the complete learning curves across every requested training size
    figure.suptitle("LASANA Learning Curves", fontsize=16)
    figure.tight_layout(rect=(0, 0.07, 1, 0.97))
    figure.savefig(FIGURE, dpi=300, bbox_inches="tight")

    # Save a second view focused on the early portion of the same learning curves
    for axis, _, _, _, _, _ in panels:
        axis.set_xlim(0, ZOOMED_PLOT_MAX_TRAIN_RUNS)

    figure.suptitle("LASANA Learning Curves — First 3,000 Training Runs", fontsize=16)
    figure.savefig(ZOOMED_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_experiment():
    experiment_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Preserve the exact active configuration alongside this run's results
    shutil.copy2(config.__file__, CONFIG_SNAPSHOT)

    data = load_data()
    tasks = prepare_tasks(data)
    results = []
    total_model_runs = len(SEEDS) * len(TRAIN_SIZES) * len(TASKS) * len(METHODS)
    completed_model_runs = 0

    print(f"Starting learning-curve experiment: {len(SEEDS)} seeds, {len(TRAIN_SIZES)} training sizes, {len(TASKS)} tasks, {len(METHODS)} methods ({total_model_runs} total model runs)", flush=True)
    print(f"Dataset: {config.RUN_NAME}/{config.DF_FILENAME}", flush=True)
    print(f"Results directory: {OUTPUT_DIR}", flush=True)
    print(f"Configuration snapshot: {CONFIG_SNAPSHOT}", flush=True)

    # Every seed uses a different fixed run-level train/validation/test split
    for seed_index, seed in enumerate(SEEDS, start=1):
        training_pool, validation_runs, test_runs = split_runs(data, seed)
        print(f"\nSeed {seed_index}/{len(SEEDS)}: {seed} ({len(training_pool)} train, {len(validation_runs)} validation, {len(test_runs)} test runs)", flush=True)

        # Larger train sizes take a superset of the smaller sized training sets
        for size_index, train_size in enumerate(TRAIN_SIZES, start=1):
            train_runs = training_pool[:train_size]
            scaler_data = data[data["Run_Number"].isin(train_runs)][config.LIST_OF_COLUMNS_X]
            scaler = StandardScaler().fit(scaler_data)
            print(f"  Training size {size_index}/{len(TRAIN_SIZES)}: {train_size} circuit runs ({len(scaler_data)} total rows)", flush=True)

            for task_index, task in enumerate(TASKS, start=1):
                task_data, target, classification = tasks[task]
                arrays = task_arrays(task_data, target, train_runs, validation_runs, test_runs)
                y_test = arrays[-1]
                print(f"    Task {task_index}/{len(TASKS)}: {TASK_LABELS[task]} ({len(arrays[0])} train, {len(arrays[2])} validation, {len(arrays[4])} test samples)", flush=True)

                # Fit all five model types on the same task-specific rows
                for method_index, method in enumerate(METHODS, start=1):
                    model_run_number = completed_model_runs + 1
                    print(f"      Method {method_index}/{len(METHODS)} [{model_run_number}/{total_model_runs} overall]: running {METHOD_LABELS[method]}...", flush=True)
                    prediction, training_time, inference_time = predict(
                        method,
                        classification,
                        arrays,
                        scaler,
                        seed,
                    )
                    metrics = calculate_metrics(y_test, prediction, classification)
                    completed_model_runs += 1

                    if classification:
                        metric_text = f"accuracy={metrics['accuracy']:.6f}, NMSE={metrics['nmse']:.6f}"
                    else:
                        metric_text = f"NMSE={metrics['nmse']:.6f}, MSE={metrics['mse']:.6g}, MAE={metrics['mae']:.6g}, MAPE={metrics['mape_percent']:.3f}%"

                    print(f"        completed: {metric_text}, training={training_time:.3f}s, inference={inference_time:.3f}s, elapsed={time.time() - experiment_start:.1f}s", flush=True)
                    results.append(
                        {
                            "method": method,
                            "train_runs": train_size,
                            "seed": seed,
                            "task": task,
                            "training_samples": len(arrays[0]),
                            "validation_samples": len(arrays[2]),
                            "test_samples": len(arrays[4]),
                            **metrics,
                            "training_time_seconds": training_time,
                            "inference_time_seconds": inference_time,
                            "total_time_seconds": training_time + inference_time,
                        }
                    )

                    # Save after every model so partial results remain visible during the run
                    pd.DataFrame(results).to_csv(RAW_RESULTS, index=False)

    plot_results(pd.DataFrame(results))
    print(f"\nCompleted all {total_model_runs} model runs.", flush=True)
    print(f"Raw results: {RAW_RESULTS}", flush=True)
    print(f"Summary results: {SUMMARY_RESULTS}", flush=True)
    print(f"Figure: {FIGURE}", flush=True)
    print(f"Zoomed figure: {ZOOMED_FIGURE}", flush=True)
    print(f"Total experiment time: {time.time() - experiment_start:.2f} seconds", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.plot_only:
        plot_results(pd.read_csv(RAW_RESULTS))
    else:
        run_experiment()


if __name__ == "__main__":
    main()
