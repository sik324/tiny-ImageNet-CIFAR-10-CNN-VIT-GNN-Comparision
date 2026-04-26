
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, train_loader, val_loader,
                num_epochs=30, learning_rate=0.001,
                device=None, save_dir="checkpoints",
                model_name="model"):
    """
    Trains the CNN model and validates after every epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    os.makedirs(save_dir, exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }
    best_val_acc = 0.0

    for epoch in range(num_epochs):

        # ── Training phase ───────────────────────────────────────
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for images, labels in tqdm(train_loader,
                                    desc=f"Epoch {epoch+1}/{num_epochs}",
                                    leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            predicted      = outputs.argmax(dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total   += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = 100 * train_correct / train_total

        # ── Validation phase ─────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(device)
                labels  = labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item()
                predicted    = outputs.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = 100 * val_correct / val_total

        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        print(f"Epoch [{epoch+1:02d}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Train Acc: {avg_train_acc:.2f}% "
              f"Val Loss: {avg_val_loss:.4f} "
              f"Val Acc: {avg_val_acc:.2f}%")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best model saved! Val Acc: {avg_val_acc:.2f}%")

    print(f"\nTraining complete!")
    print(f"Best val accuracy: {best_val_acc:.2f}%")

    return history


# ── Plot training history ─────────────────────────────────────────────
def plot_training_history(history, dataset_name,
                           save_dir="results/training"):
    """
    Plots training and validation loss and accuracy curves.

    Args:
        history      : dict returned by train_model()
        dataset_name : name for plot title and filename
        save_dir     : folder to save plot
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — {dataset_name}", fontsize=14)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"],
                 label="Train Loss", color="#562100", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],
                 label="Val Loss",   color="#C8A951", linewidth=2)
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"],
                 label="Train Acc", color="#562100", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],
                 label="Val Acc",   color="#C8A951", linewidth=2)
    axes[1].set_title("Accuracy per Epoch (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy %")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/{dataset_name}_training_history.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved to {path}")


# ── Compare both datasets training curves ────────────────────────────
def plot_comparison(cifar_history, tiny_history,
                    save_dir="results/training"):
    """
    Plots CIFAR-10 and Tiny ImageNet training curves side by side.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs_cifar = range(1, len(cifar_history["train_loss"]) + 1)
    epochs_tiny  = range(1, len(tiny_history["train_loss"])  + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CIFAR-10 vs Tiny ImageNet — Training Comparison",
                 fontsize=14)

    # CIFAR-10 loss
    axes[0,0].plot(epochs_cifar, cifar_history["train_loss"],
                   label="Train", color="#562100", linewidth=2)
    axes[0,0].plot(epochs_cifar, cifar_history["val_loss"],
                   label="Val",   color="#C8A951", linewidth=2)
    axes[0,0].set_title("CIFAR-10 — Loss")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # CIFAR-10 accuracy
    axes[0,1].plot(epochs_cifar, cifar_history["train_acc"],
                   label="Train", color="#562100", linewidth=2)
    axes[0,1].plot(epochs_cifar, cifar_history["val_acc"],
                   label="Val",   color="#C8A951", linewidth=2)
    axes[0,1].set_title("CIFAR-10 — Accuracy (%)")
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].set_ylabel("Accuracy %")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Tiny ImageNet loss
    axes[1,0].plot(epochs_tiny, tiny_history["train_loss"],
                   label="Train", color="#562100", linewidth=2)
    axes[1,0].plot(epochs_tiny, tiny_history["val_loss"],
                   label="Val",   color="#C8A951", linewidth=2)
    axes[1,0].set_title("Tiny ImageNet — Loss")
    axes[1,0].set_xlabel("Epoch")
    axes[1,0].set_ylabel("Loss")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Tiny ImageNet accuracy
    axes[1,1].plot(epochs_tiny, tiny_history["train_acc"],
                   label="Train", color="#562100", linewidth=2)
    axes[1,1].plot(epochs_tiny, tiny_history["val_acc"],
                   label="Val",   color="#C8A951", linewidth=2)
    axes[1,1].set_title("Tiny ImageNet — Accuracy (%)")
    axes[1,1].set_xlabel("Epoch")
    axes[1,1].set_ylabel("Accuracy %")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/comparison_training_curves.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved to {path}")
