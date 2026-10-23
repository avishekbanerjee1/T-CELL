import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from fcm_utils import load_rgb, fcm_membership_map, make_model_input

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True); p.add_argument("--model", required=True)
    p.add_argument("--out", default="runs/evaluation"); p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.split)/"test.csv")
    model = tf.keras.models.load_model(args.model)
    y_true, probs = [], []
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start+args.batch_size]
        xs = []
        for _, row in batch.iterrows():
            rgb = load_rgb(row.path); membership, _ = fcm_membership_map(rgb)
            xs.append(make_model_input(rgb, membership))
        pred = model.predict(np.stack(xs), verbose=0).ravel()
        probs.extend(pred.tolist()); y_true.extend(batch.label.astype(int).tolist())
    y_true = np.array(y_true); probs = np.array(probs); y_pred = (probs >= .5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) == 2 else float("nan")
    }
    pd.DataFrame([metrics]).to_csv(out/"metrics.csv", index=False)
    pd.DataFrame({"path": df.path, "actual": y_true, "malignant_probability": probs,
                  "predicted": y_pred}).to_csv(out/"predictions.csv", index=False)
    (out/"classification_report.txt").write_text(classification_report(
        y_true, y_pred, target_names=["benign", "malignant"], zero_division=0))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Benign","Malignant"])
    disp.plot(values_format="d"); plt.tight_layout(); plt.savefig(out/"confusion_matrix.png", dpi=160); plt.close()
    if len(np.unique(y_true)) == 2:
        RocCurveDisplay.from_predictions(y_true, probs); plt.tight_layout()
        plt.savefig(out/"roc_curve.png", dpi=160); plt.close()
    print(pd.Series(metrics).to_string())
    print(f"Saved evaluation outputs to {out.resolve()}")

if __name__ == "__main__":
    main()
