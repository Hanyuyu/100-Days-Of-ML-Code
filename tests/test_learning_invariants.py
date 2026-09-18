"""Regression tests for statistical correctness, not exact historical scores."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_lesson(name):
    with tempfile.TemporaryDirectory() as directory:
        previous = os.environ.get("COURSE_OUTPUT_DIR")
        os.environ["COURSE_OUTPUT_DIR"] = directory
        # The shared output directory is set at import time; reload for each run.
        import sys
        sys.modules.pop("course_utils", None)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return runpy.run_path(str(ROOT / "Code" / name))
        finally:
            if previous is None:
                os.environ.pop("COURSE_OUTPUT_DIR", None)
            else:
                os.environ["COURSE_OUTPUT_DIR"] = previous


class LearningInvariants(unittest.TestCase):
    def test_imputer_uses_only_training_rows_and_survives_unseen_country(self):
        scope = run_lesson("Day 1_Data_Preprocessing.py")
        preprocess = scope["preprocessor"]
        imputer = preprocess.named_transformers_["numeric"].named_steps["impute"]
        expected = scope["X_train"][["Age", "Salary"]].mean().to_numpy()
        np.testing.assert_allclose(imputer.statistics_, expected)
        before = imputer.statistics_.copy()
        novel = scope["X_test"].copy()
        novel["Country"] = "Previously unseen"
        novel["Age"] = 10000
        novel["Salary"] = np.nan
        transformed = preprocess.transform(novel)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape[1], scope["X_train_ready"].shape[1])
        np.testing.assert_array_equal(imputer.statistics_, before)

    def test_zero_spending_is_preserved(self):
        scope = run_lesson("Day 3_Multiple_Linear_Regression.py")
        model = scope["model"]
        zero = scope["X"].loc[scope["X"]["R&D Spend"] == 0]
        self.assertGreater(len(zero), 0)
        transformed = model.named_steps["preprocess"].transform(zero)
        names = list(model.named_steps["preprocess"].get_feature_names_out())
        self.assertTrue(np.all(transformed[:, names.index("remainder__R&D Spend")] == 0))

    def test_prediction_cannot_refit_svm_scaler(self):
        scope = run_lesson("Day 13_SVM.py")
        model = scope["model"]
        scaler = model.named_steps["scale"]
        expected = scope["X_train"].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected)
        before = scaler.mean_.copy()
        extreme = scope["X_test"].copy() * 100
        model.predict(extreme)
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_decision_plot_colors_match_class_regions(self):
        import sys
        from unittest.mock import patch
        from sklearn.tree import DecisionTreeClassifier
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(ROOT / "Code"))
        import course_utils
        X = np.array([[0., 0.], [0., 1.], [2., 0.], [2., 1.]])
        y = np.array([10, 10, 20, 20])
        model = DecisionTreeClassifier().fit(X, y)
        with patch.object(course_utils, "finish_plot"):
            course_utils.decision_plot(model, X, y, "test", ["x", "y"])
            ax = plt.gca()
            regions, points0, points1 = ax.collections
            for index, points in enumerate([points0, points1]):
                np.testing.assert_allclose(points.get_facecolors()[0, :3], regions.cmap(index)[:3])
            plt.close("all")

    def test_numpy_network_gradients_and_learning(self):
        scope = run_lesson("Day 18_Numpy_Neural_Network.py")
        self.assertLess(max(scope["errors"]), 1e-6)
        self.assertLess(scope["losses"][-1], scope["losses"][0] * 0.6)
        self.assertTrue(np.isfinite(scope["losses"]).all())
        final, _ = scope["loss_and_gradients"](scope["X_train"], scope["y_train"], scope["params"])
        self.assertAlmostEqual(scope["losses"][-1], final)
        with self.assertRaisesRegex(ValueError, "logits"):
            scope["loss_and_gradients"](scope["X_train"], scope["y_train"].ravel(), scope["params"])


if __name__ == "__main__":
    unittest.main()
