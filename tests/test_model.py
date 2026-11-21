from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from trendit_lab.model import build_models, evaluate, positive_probability


def test_pipeline_fits_statistics_only_on_training_and_ignores_extra_columns():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "configs/baseline.toml").read_text())
    config["numeric_features"] = ["T0_view_log"]
    train = pd.DataFrame({"T0_view_log": [1., 2., 3., 4.], "category_group": ["A", "B", "A", "B"],
                          "trending_duration_h": [6., 60., 6., 60.]})
    model = build_models(config)["logistic"].fit(train, [0, 1, 0, 1])
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
    np.testing.assert_allclose(scaler.mean_, [2.5])
    validation = pd.DataFrame({"T0_view_log": [1000.], "category_group": ["UNSEEN"],
                               "trending_duration_h": [999.]})
    first = positive_probability(model, validation)
    validation["trending_duration_h"] = -999
    np.testing.assert_array_equal(first, positive_probability(model, validation))
    np.testing.assert_allclose(scaler.mean_, [2.5])


def test_majority_classifier_accuracy_does_not_mean_balanced_skill():
    score = evaluate(np.array([0, 1, 1, 1]), np.array([.75, .75, .75, .75]), .5)
    assert score["accuracy"] == .75
    assert score["balanced_accuracy"] == .5
    assert score["roc_auc"] == .5
    assert score["confusion_matrix"] == [[0, 1], [0, 3]]
