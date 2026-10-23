import argparse, random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    p = argparse.ArgumentParser(description="Create stratified 70/15/15 dataset manifests.")
    p.add_argument("--data", required=True, help="Root folder containing benign/ and malignant/")
    p.add_argument("--out", default="data_split")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    root = Path(args.data)
    rows = []
    for label, folder in [(0, "benign"), (1, "malignant")]:
        class_dir = root / folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder: {class_dir}")
        for f in class_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in EXTS:
                rows.append({"path": str(f.resolve()), "label": label,
                             "class_name": folder})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No images found. Check --data and class folder names.")
    if df["label"].nunique() != 2:
        raise ValueError("Both benign and malignant classes must be present.")
    train, rest = train_test_split(df, test_size=0.30, random_state=args.seed,
                                   stratify=df["label"])
    val, test = train_test_split(rest, test_size=0.50, random_state=args.seed,
                                 stratify=rest["label"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"Images found: {len(df)}")
    print(f"Train={len(train)}, validation={len(val)}, test={len(test)}")
    print("WARNING: image-level split only. Use patient-level grouping when IDs exist.")

if __name__ == "__main__":
    main()
