import os
import argparse
import random
import numpy as np
import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class QuickDrawBinaryDataset(Dataset):

    def __init__(self, npy_path: str, max_samples: int = 10000, seed: int = 42):
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Dataset not found: {npy_path}")

        rng = np.random.default_rng(seed)
        data = np.load(npy_path)

        if data.ndim == 2 and data.shape[1] == 784:
            data = data.reshape(-1, 28, 28)
        elif data.ndim == 3 and data.shape[1:] == (28, 28):
            pass
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        data = data.astype(np.float32)

        if data.max() > 1.5:
            data = data / 255.0

        if len(data) > max_samples:
            indices = rng.choice(len(data), size=max_samples, replace=False)
            data = data[indices]

        positive = data.copy()

        negative = data.reshape(len(data), -1).copy()
        for i in range(len(negative)):
            rng.shuffle(negative[i])
        negative = negative.reshape(-1, 28, 28)

        x = np.concatenate([positive, negative], axis=0)
        y = np.concatenate([
            np.ones(len(positive), dtype=np.int64),
            np.zeros(len(negative), dtype=np.int64)
        ])

        perm = rng.permutation(len(x))
        self.x = x[perm]
        self.y = y[perm]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = torch.tensor(self.x[idx], dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img, label


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 28 -> 14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    dataset = QuickDrawBinaryDataset(
        npy_path=args.data_path,
        max_samples=args.max_samples,
        seed=args.seed
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        mlflow.set_tag("student_id", args.student_id)
        mlflow.set_tag("model_type", "SimpleCNN")
        mlflow.set_tag("dataset", "QuickDraw apple.npy binary derived")

        mlflow.log_params({
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "optimizer": "Adam"
        })

        best_val_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(args.out_dir, exist_ok=True)
                model_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save(model.state_dict(), model_path)

        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Save model with MLflow model flavor
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Also save raw weights as artifact
        mlflow.log_artifact(os.path.join(args.out_dir, "best_model.pt"))

        print(f"Best validation accuracy: {best_val_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/apple.npy")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--experiment_name", type=str, default="Assignment3_Haneen")
    parser.add_argument("--student_id", type=str, default="YOUR_ID")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
