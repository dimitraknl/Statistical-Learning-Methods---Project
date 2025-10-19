
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser(
    description="Generate CV result plots (validation curve, bar, heatmap, scatter) for SVM kernels."
)
parser.add_argument(
    "--kernel",
    type=str,
    required=True,
    choices=["RBF", "Linear", "Polynomial"],
    help="Specify which kernel's CV results to plot.",
)
args = parser.parse_args()
kernel = args.kernel

#paths
base_dir = Path("results/svm_grid_val") / kernel
csv_path = base_dir / f"cv_{kernel.lower()}.csv"
plots_dir = base_dir

#load data
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")
sns.set(style="whitegrid", font_scale=1.2)
# autofix column names
rename_map = {
    "param_svc__C": "param_C",
    "param_svc__gamma": "param_gamma",
    "param_svc__kernel": "param_kernel",
    "param_svc__degree": "param_degree",
}
df.rename(columns=rename_map, inplace=True)

print("Columns after renaming:")
print(df.columns.tolist())

# validation curve
plt.figure(figsize=(9, 6))
if "param_C" in df.columns:
    x = df["param_C"].astype(float)
else:
    x = np.arange(len(df))

y = df["mean_test_score"]
yerr = df["std_test_score"]

plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=kernel)

if "param_C" in df.columns:
    plt.xscale("log")
    plt.xlabel("C (log scale)")
else:
    plt.xlabel("Model index")

plt.ylabel("Mean CV Accuracy")
plt.title(f"Validation Curve — {kernel} Kernel")
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / f"{kernel.lower()}_validation_curve.png", dpi=300)
plt.close()

# bar plot
plt.figure(figsize=(8, 5))

top = df.nlargest(10, "mean_test_score").copy()
# Create the barplot (no manual xerr)
ax = sns.barplot(
    data=top,
    x="mean_test_score",
    y=top.index,
    orient="h",
    palette="viridis"
)


plt.errorbar(
    x=top["mean_test_score"],
    y=np.arange(len(top)),
    xerr=top["std_test_score"],
    fmt="none",
    ecolor="black",
    capsize=3
)

plt.xlabel("Mean CV Accuracy")
plt.ylabel("Parameter Combination Index")
plt.title(f"Top 10 CV Results — {kernel} Kernel")
plt.tight_layout()
plt.savefig(plots_dir / f"{kernel.lower()}_barplot.png", dpi=300)
plt.close()


# heetmap
if kernel.lower() == "rbf" and "param_gamma" in df.columns:

    rbf = df.copy()
    rbf["param_C"] = pd.to_numeric(rbf["param_C"], errors="coerce")

    # Keep gamma as string (can be 'scale', numeric, etc.)
    rbf["param_gamma"] = rbf["param_gamma"].astype(str)

    # Build pivot table with gamma as categorical index
    pivot = rbf.pivot_table(
        index="param_gamma",
        columns="param_C",
        values="mean_test_score"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        cbar_kws={"label": "Mean CV Accuracy"},
    )

    plt.xlabel("C")
    plt.ylabel("Gamma")
    plt.title("RBF Kernel — Mean CV Score Heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "rbf_heatmap.png", dpi=300)
    plt.close()


# scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="std_test_score",
    y="mean_test_score",
    s=80,
    color="steelblue",
)
plt.xlabel("Std of CV Score (Variance Proxy)")
plt.ylabel("Mean CV Score (Bias Proxy)")
plt.title(f"Bias–Variance Visualization — {kernel} Kernel")
plt.tight_layout()
plt.savefig(plots_dir / f"{kernel.lower()}_scatter.png", dpi=300)
plt.close()

print(f"✅ All plots saved to: {plots_dir.resolve()}")
