import os
from pathlib import Path
import numpy as np
import argparse
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import json
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import IncrementalPCA
from torchvision import datasets, transforms


import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJECT_ROOT / "Alzheimer_Dataset_V2"
MODEL_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 96
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_dataset(split, limit_per_class=None):
    dataset = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    take_limit = (limit_per_class is not None) and (limit_per_class > 0)
    taken_per_class = {}
    X, y = [], []

    for imgs, labels in loader:
        if take_limit:
            mask = []
            for lab in labels.cpu().numpy():
                cnt = taken_per_class.get(int(lab), 0)
                if cnt < limit_per_class:
                    mask.append(True)
                    taken_per_class[int(lab)] = cnt + 1
                else:
                    mask.append(False)
            if not any(mask):
                continue
            mask_t = torch.tensor(mask, dtype=torch.bool, device=labels.device)
            imgs = imgs[mask_t]
            labels = labels[mask_t]

        # Convert from tensor [B, C, H, W] to numpy [B, features]
        imgs = imgs.cpu()
        labels = labels.cpu()
        batch = imgs.numpy().reshape(imgs.shape[0], -1)
        X.append(batch)
        y.append(labels.numpy())

    if not X:
        raise RuntimeError(f"No samples loaded from split='{split}'. Check paths/limits.")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y, dataset.classes

def _parse_gamma(val):
    try:
        return float(val)
    except ValueError:
        if val not in ("scale", "auto"):
            raise ValueError("gamma must be 'scale', 'auto', or a float string (e.g., '0.01').")
        return val

def _parse_list_of_floats(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_list_of_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _parse_list_of_strings(s):
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_list_of_gammas(s):
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("scale", "auto"):
            vals.append(tok)
        else:
            vals.append(float(tok))
    return vals

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM training for Alzheimer MRI (single-run or 5-fold CV grid-search).")
    # small run
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="If set, load at most N images per class for train/val (speeds up quick tests).")


    parser.add_argument("--include-val-in-selection", action="store_true",
                        help="If set, perform CV model selection on TRAIN+VAL; otherwise TRAIN only.")

    # grid search ranges
    parser.add_argument("--kernels", default=None,
                        help="Comma-separated kernels for grid, e.g. 'rbf,linear,poly'. If omitted, defaults to ['rbf'].")
    parser.add_argument("--Cs", default=None,
                        help="Comma-separated C values, e.g. '0.1,1,3,10'. If omitted, defaults to [0.1,1,3,10,30,100].")
    parser.add_argument("--gammas", default=None,
                        help="Comma-separated gamma values, e.g. 'scale,0.01,0.001'. If omitted, defaults to ['scale',1e-3,1e-2,1e-1].")
    parser.add_argument("--degrees", default=None,
                        help="Comma-separated polynomial degrees, e.g. '2,3'. If omitted, defaults to [2, 3].")

    # --- Outputs for grid-search artifacts (project-relative defaults) ---
    parser.add_argument("--cv-results-csv",
                        default=str((RESULTS_DIR / "svm_grid" / "cv_results.csv").resolve()),
                        help="Path to save the full CV table (all combinations with mean/std scores).")
    parser.add_argument("--best-model-out",
                        default=str((MODEL_DIR / "svm_best.pkl").resolve()),
                        help="Path to save the refit best model (pipeline = StandardScaler + best SVC).")
    parser.add_argument("--best-meta-out",
                        default=str((MODEL_DIR / "svm_best.json").resolve()),
                        help="Path to save best params/score/classes JSON.")
    args = parser.parse_args()


    for _p in [args.cv_results_csv, args.best_model_out, args.best_meta_out]:
        out_dir = os.path.dirname(_p)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    X_train, y_train, classes = load_dataset("train", limit_per_class=args.limit_per_class)
    X_val, y_val, _ = load_dataset("val", limit_per_class=args.limit_per_class)

    if args.include_val_in_selection:
        X_sel = np.vstack([X_train, X_val])
        y_sel = np.concatenate([y_train, y_val])
        print("Model selection on: TRAIN + VAL")
    else:
        X_sel, y_sel = X_train, y_train
        print("Model selection on: TRAIN only")

    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")

    # scale and train
    print("\nGrid search with 5-fold stratified CV (scoring = macro-F1)")
    os.makedirs(os.path.dirname(args.cv_results_csv), exist_ok=True)


    pipe = make_pipeline(
        IncrementalPCA(n_components=256, batch_size=256),
        StandardScaler(),
        svm.SVC()
    )

        # Param grid
    grid = []
    kernels = _parse_list_of_strings(args.kernels) if args.kernels else ["rbf"]
    Cs      = _parse_list_of_floats(args.Cs)       if args.Cs      else [0.1, 1, 3, 10, 30, 100]
    gammas  = _parse_list_of_gammas(args.gammas)   if args.gammas  else ["scale", 1e-3, 1e-2, 1e-1]
    degrees = _parse_list_of_ints(args.degrees)    if args.degrees else [2, 3]

    if "rbf" in kernels:
        grid.append({"svc__kernel": ["rbf"], "svc__C": Cs, "svc__gamma": gammas})
    if "linear" in kernels:
        grid.append({"svc__kernel": ["linear"], "svc__C": Cs})
    if "poly" in kernels:
        grid.append({"svc__kernel": ["poly"], "svc__C": Cs, "svc__gamma": gammas, "svc__degree": degrees})

    scorer = make_scorer(f1_score, average="macro")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            verbose=1
    )
    gs.fit(X_sel, y_sel)

    #save
    joblib.dump(gs.best_estimator_, args.best_model_out)
    pd.DataFrame(gs.cv_results_).to_csv(args.cv_results_csv, index=False)
    meta = {
            "classes": classes,
            "best_params": gs.best_params_,
            "best_cv_macro_f1": float(gs.best_score_),
            "selection_on": "train+val" if args.include_val_in_selection else "train_only",
            "cv_folds": 5,
    }
    with open(args.best_meta_out, "w") as f:
        json.dump(meta, f, indent=2)

        print("\nBest params:", gs.best_params_)
        print("Best CV macro-F1:", gs.best_score_)
        print(f"Refit best model saved to: {args.best_model_out}")
        print(f"Full CV table saved to: {args.cv_results_csv}")
        print(f"Meta saved to: {args.best_meta_out}")
