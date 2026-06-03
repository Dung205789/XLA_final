"""Wrapper around the provided evaluate_predictions.py evaluator."""

import json
import os
import subprocess
import sys
import tempfile


def compute_map(
    predictions: list,
    gt_json_path: str,
    eval_script: str = "public/tools/evaluate_predictions.py",
) -> tuple:
    """
    Write predictions to a temp file and call the evaluator.
    Returns: (mAP_50 float, full_score_dict)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as pf:
        json.dump(predictions, pf)
        pred_path = pf.name

    score_path = pred_path + "_score.json"
    try:
        result = subprocess.run(
            [
                sys.executable, eval_script,
                "--ground_truth", gt_json_path,
                "--predictions", pred_path,
                "--output", score_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[metrics] evaluator stderr:\n{result.stderr[:500]}")
            return 0.0, {}

        with open(score_path, encoding="utf-8") as f:
            score = json.load(f)
        return float(score.get("mAP@0.5", 0.0)), score
    finally:
        os.unlink(pred_path)
        if os.path.exists(score_path):
            os.unlink(score_path)
