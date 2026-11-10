"""One synthetic SEIR example for three parameter estimation procedures.

The default random state comes from the clock. A run records it together with
the complete configuration; --replay reproduces that recorded computation.
This module contains no downloads and uses no external epidemiological data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


PARAMETERS = ("beta", "sigma", "gamma")
METHODS = ("Genetic algorithm", "Inverse MLP", "GP calibration")
COLOURS = ("#0072B2", "#D55E00", "#009E73")


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    model = cfg["model"]
    bounds = np.asarray(model["parameter_bounds"], dtype=float)
    truth = np.asarray(model["true_parameters"], dtype=float)
    if bounds.shape != (3, 2) or np.any(bounds <= 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Provide three positive, ordered parameter intervals.")
    if truth.shape != (3,) or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1]):
        raise ValueError("The generating parameters must lie in the parameter domain.")
    initial = [model[k] for k in ("initial_exposed", "initial_infectious", "initial_removed")]
    if min(initial) < 0 or sum(initial) >= model["population"]:
        raise ValueError("Initial compartment counts must define a positive susceptible population.")
    if not (0 < model["reporting_fraction"] <= 1) or model["dispersion"] <= 0:
        raise ValueError("Invalid observation parameters.")
    if int(model["days"]) != model["days"] or model["days"] < 2:
        raise ValueError("days must be an integer of at least two.")
    ga = cfg["genetic_algorithm"]
    if not (1 <= ga["elite_count"] < ga["population_size"]):
        raise ValueError("The elite count must be smaller than the population.")


def from_unit(unit: np.ndarray, model: dict) -> np.ndarray:
    bounds = np.asarray(model["parameter_bounds"])
    return bounds[:, 0] + np.asarray(unit) * (bounds[:, 1] - bounds[:, 0])


def to_unit(parameters: np.ndarray, model: dict) -> np.ndarray:
    bounds = np.asarray(model["parameter_bounds"])
    return (np.asarray(parameters) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def seir(parameters: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return daily expected reported S->E transitions and daily S,E,I,R counts.

    Fractions are integrated for stable tolerances. A fifth state accumulates
    new infections, so daily incidence is an interval integral, not I(t).
    """
    beta, sigma, gamma = np.asarray(parameters, dtype=float)
    if np.any(np.asarray(parameters) < 0) or not np.all(np.isfinite(parameters)):
        raise ValueError("Rates must be finite and nonnegative.")
    population = model["population"]
    e0, i0, r0 = (model[k] / population for k in
                  ("initial_exposed", "initial_infectious", "initial_removed"))

    def rhs(_time: float, state: np.ndarray) -> list[float]:
        susceptible, exposed, infectious, _removed, _cumulative = state
        infections = beta * susceptible * infectious
        progressions = sigma * exposed
        removals = gamma * infectious
        return [-infections, infections - progressions,
                progressions - removals, removals, infections]

    solution = solve_ivp(
        rhs, (0, model["days"]), [1 - e0 - i0 - r0, e0, i0, r0, 0],
        t_eval=np.arange(model["days"] + 1), method="DOP853",
        rtol=model["ode_rtol"], atol=model["ode_atol"],
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    incidence = population * np.diff(solution.y[4])
    if incidence.min() < -1e-5 or solution.y[:4].min() < -1e-7:
        raise ArithmeticError("The solver returned invalid compartment counts or incidence.")
    mean = model["reporting_fraction"] * np.maximum(incidence, 0)
    return mean, population * solution.y[:4].T


def observe(mean: np.ndarray, dispersion: float, rng: np.random.Generator) -> np.ndarray:
    """Negative binomial counts with variance mean + mean**2 / dispersion."""
    return rng.negative_binomial(dispersion, dispersion / (dispersion + mean))


def negative_log_likelihood(counts: np.ndarray, mean: np.ndarray, dispersion: float) -> float:
    counts, mean = np.asarray(counts), np.asarray(mean)
    if counts.shape != mean.shape or np.any(counts < 0) or np.any(mean < 0):
        raise ValueError("Counts and means must have equal shapes and be nonnegative.")
    if np.any(counts != np.floor(counts)) or dispersion <= 0:
        raise ValueError("Counts must be integers and dispersion must be positive.")
    log_probability = (
        gammaln(counts + dispersion) - gammaln(dispersion) - gammaln(counts + 1)
        + dispersion * (np.log(dispersion) - np.log(dispersion + mean))
        + xlogy(counts, mean) - counts * np.log(dispersion + mean)
    )
    return float(-log_probability.sum())


class Objective:
    """Count actual SEIR evaluations of the fixed observation series."""

    def __init__(self, counts: np.ndarray, model: dict):
        self.counts = counts
        self.model = model
        self.calls = 0

    def __call__(self, unit: np.ndarray) -> float:
        mean, _ = seir(from_unit(unit, self.model), self.model)
        self.calls += 1
        return negative_log_likelihood(self.counts, mean, self.model["dispersion"])


def genetic_algorithm(counts: np.ndarray, cfg: dict, random_state: int) -> dict:
    model, settings = cfg["model"], cfg["genetic_algorithm"]
    rng = np.random.default_rng(random_state)
    objective = Objective(counts, model)
    size, elite = settings["population_size"], settings["elite_count"]
    population = qmc.LatinHypercube(3, seed=random_state).random(size)
    losses = np.array([objective(candidate) for candidate in population])
    history = [{"generation": 0, "evaluations": objective.calls, "best_nll": float(losses.min())}]
    for generation in range(1, settings["generations"] + 1):
        selected = np.argsort(losses)[:elite]
        children = []
        fraction = (generation - 1) / max(settings["generations"] - 1, 1)
        scale = ((1 - fraction) * settings["initial_mutation_scale"]
                 + fraction * settings["final_mutation_scale"])
        for _ in range(size - elite):
            parents = []
            for _parent in range(2):
                tournament = rng.integers(size, size=settings["tournament_size"])
                parents.append(population[tournament[np.argmin(losses[tournament])]])
            blend = rng.uniform(-0.1, 1.1, size=3)
            child = blend * parents[0] + (1 - blend) * parents[1]
            mutation = rng.random(3) < settings["mutation_probability"]
            child += mutation * rng.normal(0, scale, 3)
            children.append(np.clip(child, 0, 1))
        children = np.asarray(children)
        population = np.vstack((population[selected], children))
        losses = np.r_[losses[selected], [objective(child) for child in children]]
        history.append({"generation": generation, "evaluations": objective.calls,
                        "best_nll": float(losses.min())})
    best = int(np.argmin(losses))
    return {"parameters": from_unit(population[best], model), "history": history,
            "simulator_evaluations": objective.calls}


def training_pairs(number: int, copies: int, model: dict, design_state: int,
                   observation_state: int) -> tuple[np.ndarray, np.ndarray]:
    unit = qmc.LatinHypercube(3, seed=design_state).random(number)
    rng = np.random.default_rng(observation_state)
    features = np.empty((number * copies, model["days"]))
    for index, parameters in enumerate(from_unit(unit, model)):
        mean, _ = seir(parameters, model)
        for copy_index in range(copies):
            features[index * copies + copy_index] = observe(mean, model["dispersion"], rng)
    return features, np.repeat(unit, copies, axis=0)


def inverse_mlp(counts: np.ndarray, cfg: dict, states: dict) -> dict:
    model, settings = cfg["model"], cfg["inverse_mlp"]
    copies = settings["observations_per_parameter"]
    print("  Generating independent MLP training and validation simulations...", flush=True)
    train_x, train_y = training_pairs(settings["training_parameters"], copies, model,
                                    states["training_design"], states["training_noise"])
    validation_x, validation_y = training_pairs(settings["validation_parameters"], copies, model,
                                              states["validation_design"], states["validation_noise"])
    scaler = StandardScaler().fit(np.log1p(train_x))
    train_x = scaler.transform(np.log1p(train_x))
    validation_x = scaler.transform(np.log1p(validation_x))
    network = MLPRegressor(
        hidden_layer_sizes=tuple(settings["hidden_layers"]), activation=settings["activation"],
        solver="adam", batch_size=settings["batch_size"], alpha=settings["alpha"],
        learning_rate_init=settings["learning_rate"], random_state=states["mlp"], shuffle=True,
    )
    best_model, best_loss, patience_loss, waiting = None, np.inf, np.inf, 0
    best_epoch, history = 0, []
    for epoch in range(1, settings["maximum_epochs"] + 1):
        network.partial_fit(train_x, train_y)
        loss = float(np.mean((network.predict(validation_x) - validation_y) ** 2))
        history.append({"epoch": epoch, "training_objective": float(network.loss_), "validation_mse": loss})
        if loss < best_loss:
            best_model, best_loss, best_epoch = copy.deepcopy(network), loss, epoch
        if loss < patience_loss - settings["minimum_improvement"]:
            patience_loss, waiting = loss, 0
        else:
            waiting += 1
        if waiting >= settings["patience"]:
            break
    # The evaluation counts first enter the estimator here, after model selection.
    raw = best_model.predict(scaler.transform(np.log1p(counts)[None, :]))[0]
    bounded = np.clip(raw, 0, 1)
    return {"parameters": from_unit(bounded, model), "history": history,
            "simulator_evaluations": settings["training_parameters"] + settings["validation_parameters"],
            "training_examples": len(train_x), "validation_examples": len(validation_x),
            "best_epoch": best_epoch, "validation_mse": best_loss,
            "raw_parameters": from_unit(raw, model).tolist(),
            "projected_to_bounds": bool(np.any(raw != bounded))}


def surrogate_calibration(counts: np.ndarray, cfg: dict, random_state: int) -> dict:
    model, settings = cfg["model"], cfg["surrogate"]
    rng = np.random.default_rng(random_state)
    objective = Objective(counts, model)
    design = qmc.LatinHypercube(3, seed=random_state).random(settings["initial_design"])
    losses = np.array([objective(candidate) for candidate in design])
    kernel = ConstantKernel(1, (0.01, 100)) * Matern(np.full(3, 0.25), (0.03, 3), nu=2.5)
    history, gp_messages = [], set()
    for iteration in range(settings["iterations"] + 1):
        optimise = iteration % settings["hyperparameter_interval"] == 0
        process = GaussianProcessRegressor(
            kernel=kernel, alpha=settings["gp_alpha"], normalize_y=True,
            optimizer="fmin_l_bfgs_b" if optimise else None,
            n_restarts_optimizer=0, random_state=random_state,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            process.fit(design, losses / model["days"])
        gp_messages.update(str(w.message) for w in caught)
        kernel = process.kernel_
        pool = rng.random((settings["candidate_pool"], 3))
        incumbent = design[np.argmin(losses)]
        local = np.clip(incumbent + rng.normal(0, 0.08, (settings["local_candidates"], 3)), 0, 1)
        pool = np.vstack((pool, local, incumbent))
        weight = 0 if iteration == settings["iterations"] else settings["lcb_weight"]

        def acquisition(candidates: np.ndarray) -> np.ndarray:
            mean, standard_deviation = process.predict(np.atleast_2d(candidates), return_std=True)
            return mean - weight * standard_deviation

        pool_scores = acquisition(pool)
        starts = pool[np.argsort(pool_scores)[:4]]
        options = [starts[0]]
        for start in starts:
            solution = minimize(lambda x: float(acquisition(x)[0]), start,
                                bounds=[(0, 1)] * 3, method="L-BFGS-B",
                                options={"maxiter": 60})
            if np.all(np.isfinite(solution.x)):
                options.append(np.clip(solution.x, 0, 1))
        candidate = options[int(np.argmin(acquisition(np.asarray(options))))]
        distances = np.linalg.norm(design - candidate, axis=1)
        if distances.min() < 1e-6:
            for index in np.argsort(pool_scores):
                if np.min(np.linalg.norm(design - pool[index], axis=1)) >= 1e-6:
                    candidate = pool[index]
                    break
        actual_loss = objective(candidate)
        design = np.vstack((design, candidate))
        losses = np.r_[losses, actual_loss]
        history.append({"iteration": iteration + 1, "evaluations": objective.calls,
                        "candidate_nll": actual_loss, "best_nll": float(losses.min()),
                        "final_mean_check": iteration == settings["iterations"]})
    best = int(np.argmin(losses))
    return {"parameters": from_unit(design[best], model), "history": history,
            "simulator_evaluations": objective.calls, "gp_messages": sorted(gp_messages)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(output: Path, counts: np.ndarray, mean: np.ndarray, results: dict, model: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.edgecolor": "black", "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.65, 1]})
    days = np.arange(1, model["days"] + 1)
    axes[0].scatter(days, counts, s=9, color="#777777", alpha=0.5, label="Synthetic counts", zorder=2)
    axes[0].plot(days, mean, color="#222222", linestyle="--", linewidth=1.6, label="Generating model")
    offsets = (-0.18, 0, 0.18)
    for method, colour, offset in zip(METHODS, COLOURS, offsets):
        axes[0].plot(days, results[method]["mean"], color=colour, linewidth=1.3, label=method)
        axes[1].scatter(results[method]["parameters"], np.arange(3) + offset,
                        color=colour, s=28, marker="o", zorder=3)
    axes[1].scatter(model["true_parameters"], np.arange(3), marker="|", color="#222222", s=160, zorder=4)
    axes[0].set(xlabel="Day", ylabel="Daily reported infections", xlim=(1, model["days"]))
    axes[1].set(yticks=np.arange(3), yticklabels=[r"$\beta$", r"$\sigma$", r"$\gamma$"],
                xlabel=r"Rate (day$^{-1}$)", ylim=(2.5, -0.5))
    axes[0].set_title("(a) Observations and fitted means", loc="left", fontsize=10)
    axes[1].set_title("(b) Rate estimates", loc="left", fontsize=10)
    for axis in axes:
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.28, top=0.89, wspace=0.36)
    fig.savefig(output / "seir_comparison.png", dpi=300)
    fig.savefig(output / "seir_comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def run(cfg: dict, output: Path, clock_seed: int) -> dict:
    names = ("evaluation", "ga", "training_design", "training_noise", "validation_design",
             "validation_noise", "mlp", "gp")
    streams = np.random.SeedSequence(clock_seed).spawn(len(names))
    states = {name: int(stream.generate_state(1)[0]) for name, stream in zip(names, streams)}
    model = cfg["model"]
    truth = np.asarray(model["true_parameters"])
    mean, _ = seir(truth, model)
    counts = observe(mean, model["dispersion"], np.random.default_rng(states["evaluation"]))
    results = {}
    for label, fit in zip(METHODS, (
        lambda: genetic_algorithm(counts, cfg, states["ga"]),
        lambda: inverse_mlp(counts, cfg, states),
        lambda: surrogate_calibration(counts, cfg, states["gp"]),
    )):
        print(f"Running {label}...", flush=True)
        result = fit()
        fitted, _ = seir(result["parameters"], model)
        result["mean"] = fitted
        result["nll"] = negative_log_likelihood(counts, fitted, model["dispersion"])
        result["mean_curve_rmse"] = float(np.sqrt(np.mean((fitted - mean) ** 2)))
        result["relative_parameter_error"] = ((result["parameters"] - truth) / truth).tolist()
        results[label] = result
        print(f"  rates={result['parameters']}, negative log likelihood={result['nll']:.3f}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    synthetic = [{"day": day + 1, "expected_count": float(mean[day]), "observed_count": int(counts[day])}
                 for day in range(model["days"])]
    write_csv(output / "synthetic_data.csv", synthetic)
    fitted_rows = []
    for day in range(model["days"]):
        row = {"day": day + 1}
        row.update({label: float(results[label]["mean"][day]) for label in METHODS})
        fitted_rows.append(row)
    write_csv(output / "fitted_curves.csv", fitted_rows)
    summary = [{"method": "Generating parameters", **dict(zip(PARAMETERS, truth)),
                "negative_log_likelihood": negative_log_likelihood(counts, mean, model["dispersion"]),
                "mean_curve_rmse": 0.0, "simulator_evaluations": 1}]
    for label, filename in zip(METHODS, ("ga", "mlp", "gp")):
        result = results[label]
        summary.append({"method": label, **dict(zip(PARAMETERS, result["parameters"])),
                        "negative_log_likelihood": result["nll"], "mean_curve_rmse": result["mean_curve_rmse"],
                        "simulator_evaluations": result["simulator_evaluations"]})
        write_csv(output / f"{filename}_history.csv", result["history"])
    write_csv(output / "estimates.csv", summary)
    make_figure(output, counts, mean, results, model)
    details = {}
    for label, result in results.items():
        details[label] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                          for key, value in result.items() if key not in ("mean", "history")}
    metadata = {"configuration": cfg, "clock_seed": clock_seed, "random_states": states,
                "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "scikit-learn", "matplotlib", "threadpoolctl")},
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "results": details,
                "scope": "One synthetic illustration; no repeated performance study or timing comparison.",
                "counting": "Method evaluations include training/validation designs; final reconstruction adds one SEIR solve per method."}
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--replay", type=Path, help="Use the configuration and random state recorded in a run.json file.")
    args = parser.parse_args()
    if args.replay:
        previous = json.loads(args.replay.read_text(encoding="utf-8"))
        cfg, clock_seed = previous["configuration"], previous["clock_seed"]
        validate_config(cfg)
        current_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if current_hash != previous["source_sha256"]:
            raise ValueError("The source differs from the recorded run; use the matching study.py version.")
    else:
        cfg, clock_seed = read_config(args.config), time.time_ns()
    with threadpool_limits(limits=1):
        run(cfg, args.output, clock_seed)
    print(f"Results saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
