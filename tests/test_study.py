"""Checks of the model, observation definition, likelihood and optimiser."""

import copy
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study


class StudyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = study.read_config(Path(study.__file__).with_name("config.json"))

    def test_population_conservation_and_incidence_definition(self):
        model = self.cfg["model"]
        mean, states = study.seir(model["true_parameters"], model)
        self.assertEqual(len(mean), model["days"])
        self.assertGreaterEqual(states.min(), -1e-5)
        np.testing.assert_allclose(states.sum(axis=1), model["population"], rtol=1e-11)
        expected_from_susceptibles = -model["reporting_fraction"] * np.diff(states[:, 0])
        np.testing.assert_allclose(mean, expected_from_susceptibles, atol=1e-8, rtol=1e-8)

    def test_no_transmission_means_no_new_infections(self):
        # Infectious people remain present: this would fail if I(t) were used as incidence.
        mean, states = study.seir([0, 0.2, 0.1], self.cfg["model"])
        np.testing.assert_allclose(mean, 0, atol=1e-12)
        self.assertGreater(states[:, 2].max(), 0)

    def test_likelihood_matches_independent_library_parameterisation(self):
        counts = np.array([0, 1, 5, 23, 200])
        mean = np.array([0, 0.3, 8, 20, 150])
        dispersion = 7.5
        reference = -nbinom.logpmf(counts, dispersion, dispersion / (dispersion + mean)).sum()
        calculated = study.negative_log_likelihood(counts, mean, dispersion)
        self.assertAlmostEqual(calculated, reference, places=11)
        self.assertTrue(np.isinf(study.negative_log_likelihood(np.array([1]), np.array([0]), dispersion)))

    def test_numerical_tolerances_are_sufficient(self):
        model = copy.deepcopy(self.cfg["model"])
        tight = dict(model, ode_rtol=1e-11, ode_atol=1e-13)
        bounds = np.asarray(model["parameter_bounds"])
        for parameters in (model["true_parameters"], bounds[:, 0], bounds[:, 1]):
            mean, _ = study.seir(parameters, model)
            reference, _ = study.seir(parameters, tight)
            self.assertLess(np.max(np.abs(mean - reference)) / max(1, reference.max()), 2e-6)

    def test_genetic_search_preserves_best_candidate_and_bounds(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["genetic_algorithm"].update(population_size=12, elite_count=2, generations=4)
        mean, _ = study.seir(cfg["model"]["true_parameters"], cfg["model"])
        counts = np.round(mean).astype(int)
        state = time.time_ns() % (2 ** 32)
        result = study.genetic_algorithm(counts, cfg, state)
        best = np.array([row["best_nll"] for row in result["history"]])
        self.assertTrue(np.all(np.diff(best) <= 0))
        unit = study.to_unit(result["parameters"], cfg["model"])
        self.assertTrue(np.all((unit >= 0) & (unit <= 1)))
        self.assertEqual(result["simulator_evaluations"], 12 + 4 * 10)


if __name__ == "__main__":
    unittest.main()
