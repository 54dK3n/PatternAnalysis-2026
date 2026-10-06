"""Check train-only PCA and protected patient targets with real fixed-head fits."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation.feature_probes import run_feature_probes, training_pca
import test_feature_probes as fixtures


class FollowupProbeTests(unittest.TestCase):
    """Exercise deterministic projection without using early-stop fitting data."""

    def setUp(self) -> None:
        """Use a small nonconstant synthetic cohort and independent output paths."""
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_training_pca_is_unwhitened_and_preserves_full_rank_pair_distances(self) -> None:
        """A complete orthonormal basis preserves distances in an analytic example."""
        train = torch.tensor([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]])
        before = train.clone()
        projected, early, metadata = training_pca(train, train[:1], 2)
        self.assertTrue(torch.allclose(torch.cdist(train, train), torch.cdist(projected, projected), atol=1e-6))
        self.assertTrue(torch.equal(train, before))
        self.assertFalse(metadata["whiten"])
        self.assertEqual(metadata["numerical_rank"], 2)
        self.assertTrue(torch.allclose(early, projected[:1]))
        self.assertEqual(metadata["fit_role"], "train")

    def test_early_features_and_labels_cannot_change_pca_or_fitted_training_outputs(self) -> None:
        """Real fitting outcomes must stay identical under extreme early data changes."""
        values = fixtures._fixture()
        baseline = run_feature_probes(*values, self.root / "original", epochs=3, pca_components=1)
        modified = copy.deepcopy(values)
        modified[2]["stage1"] += 1000
        for row in modified[3]:
            row["label"] = 1 - row["label"]
        shifted = run_feature_probes(*modified, self.root / "changed", epochs=3, pca_components=1)
        self.assertEqual(baseline["stages"]["stage1"]["pca"], shifted["stages"]["stage1"]["pca"])
        self.assertEqual(baseline["stages"]["stage1"]["train"], shifted["stages"]["stage1"]["train"])
        self.assertEqual(baseline["stages"]["stage1"]["feature_dim"], 1)
        self.assertEqual(baseline["feature_projection"], "train_only_unwhitened_PCA")

    def test_rank_deficiency_is_recorded_without_dimension_search(self) -> None:
        """Constant directions stay declared instead of triggering held-out tuning."""
        train = torch.zeros(4, 3)
        projected, _, metadata = training_pca(train, torch.ones(2, 3), 2)
        self.assertEqual(projected.shape, (4, 2))
        self.assertEqual(metadata["numerical_rank"], 0)
        self.assertTrue(metadata["rank_deficient"])

    def test_pca_rank_limits_fail_before_publishing_metrics(self) -> None:
        """Unsupported components do not create successful probe output files."""
        for components in (0, True, 3):
            with self.assertRaises(ValueError):
                run_feature_probes(*fixtures._fixture(), self.root / "bad", epochs=1, pca_components=components)
        self.assertFalse((self.root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
