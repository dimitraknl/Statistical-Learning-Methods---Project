import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt                ### >>> added
from sklearn.metrics import classification_report, confusion_matrix, f1_score   ### >>> added

def get_dataloaders(data_root, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=transform)
    val_ds = datasets.ImageFolder(f"{data_root}/val", transform=transform)
    test_ds = datasets.ImageFolder(f"{data_root}/test", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_ds.classes



class TinyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*28*28, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    @staticmethod
    def test_model_dimensions(model, device):
        """Test that model can handle expected input sizes"""
        test_input = torch.randn(2, 3, 224, 224).to(device)
        try:
            output = model(test_input)
            print(f"Model test passed. Output shape: {output.shape}")
        except Exception as e:
            print(f"Model dimension error: {e}")


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return running_loss/total, correct/total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    val_f1 = f1_score(y_true, y_pred, average='macro')
    return running_loss/total, correct/total, val_f1


def classification_metrics(model, loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = outputs.max(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="models/cnn_baseline.pth")
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(args.data_root, args.batch_size)

    model = TinyCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0
    patience_counter = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ### >>> added: histories for plotting
    train_loss_history, val_loss_history, val_f1_history = [], [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), args.out)
            print(f"New best model saved (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # load best model
    model.load_state_dict(torch.load(args.out))
    print(f"\nBest model loaded for testing (val_acc: {best_val_acc:.4f})")

    # final evaluation on test set
    classification_metrics(model, test_loader, device, class_names)

    epochs = range(1, len(val_f1_history) + 1)
    plt.figure()
    plt.plot(epochs, val_loss_history, label='Val Loss')
    plt.plot(epochs, val_f1_history, label='Val Macro-F1')
    plt.title('Validation Loss and Macro-F1 per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/val_loss_f1_curve.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
