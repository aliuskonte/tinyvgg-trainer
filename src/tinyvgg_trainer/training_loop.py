import torch
from torch import nn, utils, optim
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model: nn.Module,
               dataloader: utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: optim.Optimizer,
               device=DEVICE):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = y_pred.softmax(dim=1).argmax(dim=1)
        train_acc += (preds == y).float().mean().item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def eval_step(model: nn.Module,
              dataloader: utils.data.DataLoader,
              loss_fn: nn.Module,
              device=DEVICE):
    """
    Функция для оценки модели на данных. Тестирование и валидация
    """

    model.eval()
    loss_sum, acc_sum = 0.0, 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss_sum += loss.item()
            preds = logits.softmax(dim=1).argmax(dim=1)
            acc_sum += (preds == y).float().mean().item()

    loss_avg = loss_sum / len(dataloader)
    acc_avg = acc_sum / len(dataloader)
    return loss_avg, acc_avg


def train(model: nn.Module,
          train_dataloader: utils.data.DataLoader,
          val_dataloader: utils.data.DataLoader,
          test_dataloader: utils.data.DataLoader,
          optimizer: optim.Optimizer,
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          save_path: str = 'saved_weights.pt'
          ):
    results = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "test_loss":  [], "test_acc":  []
    }

    best_val = float("inf")

    for epoch in tqdm(range(epochs), desc="Epoch"):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
        val_loss, val_acc = eval_step(model, val_dataloader,   loss_fn)
        test_loss, test_acc = eval_step(model, test_dataloader,  loss_fn)

        # Save best model only if val_loss improved
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

        # 4) Log to console
        print(
            f"Epoch {epoch+1}/{epochs} — "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val:   loss={val_loss:.4f},   acc={val_acc:.4f}   | "
            f"Test:  loss={test_loss:.4f},  acc={test_acc:.4f}"
        )

        # 5) Store metrics
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results