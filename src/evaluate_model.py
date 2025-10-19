import argparse
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    cohen_kappa_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
from train_cnn import TinyCNN


# === CONFUSION MATRIX ===
def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png", model_label="Model"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.title(f'{model_label} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# class metrics (Precision,Recall,F1)
def plot_class_metrics(y_true, y_pred, class_names, save_path="class_metrics.png", model_label="Model"):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = ['precision', 'recall', 'f1-score']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    plt.subplots_adjust(top=0.85)

    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in class_names]
        bars = axes[i].bar(class_names, values, color=colors[i], alpha=0.8)
        axes[i].set_title(f'{metric.capitalize()} per Class', fontweight='bold')
        axes[i].set_ylim(0, 1.1)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.suptitle(f'{model_label} - Per-Class Performance Metrics', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# roc curves
def plot_roc_curves(y_true, y_prob, class_names, save_path="roc_curves.png", model_label="Model"):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    y_prob = np.array(y_prob)

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random chance')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'{model_label} - ROC Curves per Class', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# additional metrics
def calculate_additional_metrics(y_true, y_pred, class_names):
    overall_accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"\n=== Additional Metrics ===")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    print(f"\n=== Per-Class Accuracy ===")
    for i, class_name in enumerate(class_names):
        mask = np.array(y_true) == i
        if np.sum(mask) > 0:
            acc = np.mean(np.array(y_pred)[mask] == i)
            print(f"{class_name}: {acc:.4f}")

    return overall_accuracy, balanced_acc, kappa



def evaluate_model(model_path, data_root, split="test", batch_size=32, model_type="cnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_label = model_type.upper()

    # --- Transforms ---
    if model_type == "cnn":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])

    dataset = datasets.ImageFolder(f"{data_root}/{split}", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    class_names = dataset.classes

    print(f"🔍 Evaluating model: {model_path}")
    print(f"📊 Dataset: {split} split ({len(dataset)} samples)")
    print(f"🎯 Classes: {class_names}")


    if model_type == "cnn":
        model = TinyCNN(num_classes=len(class_names)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        model = joblib.load(model_path)

    y_true, y_pred, y_prob = [], [], []


    if model_type == "cnn":
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())


    else:
        for X, y in loader:
            X_np = X.view(X.size(0), -1).numpy()
            preds = model.predict(X_np)

            # ROC support: use decision_function if available
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_np)
                if scores.ndim == 1:  # binary fix
                    scores = np.vstack([-scores, scores]).T
                y_prob.extend(scores)
            else:
                y_prob.extend(np.zeros((len(preds), len(model.classes_))))

            y_true.extend(y.numpy())
            y_pred.extend(preds)


    print("\n" + "=" * 50)
    print("📈 CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("\n" + "=" * 50)
    print("🎯 ADDITIONAL METRICS")
    print("=" * 50)
    calculate_additional_metrics(y_true, y_pred, class_names)

    print("\n" + "=" * 50)
    print("📊 GENERATING PLOTS")
    print("=" * 50)
    plot_confusion_matrix(y_true, y_pred, class_names, f"{model_type.lower()}_confusion_matrix.png", model_label)
    plot_class_metrics(y_true, y_pred, class_names, f"{model_type.lower()}_class_metrics.png", model_label)

    # --- ROC Curves ---
    if y_prob and np.array(y_prob).ndim == 2:
        plot_roc_curves(y_true, y_prob, class_names, f"{model_type.lower()}_roc_curves.png", model_label)

    print(f"\n✅ Evaluation complete! Plots saved as:")
    print(f"   - {model_type.lower()}_confusion_matrix.png")
    print(f"   - {model_type.lower()}_class_metrics.png")
    print(f"   - {model_type.lower()}_roc_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CNN or SVM model')
    parser.add_argument('--model-type', choices=['cnn', 'svm'], required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_root, args.split, args.batch_size, args.model_type)
