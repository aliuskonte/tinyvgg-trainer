
from contextlib import nullcontext
import torch
from torch import nn, utils, optim
from tqdm.auto import tqdm
from clearml import Task
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score

# ------------------------------------------------------------------
#   Глобальные оптимизации вычислений
# ------------------------------------------------------------------
# ← Изменено: включаем cudnn benchmark (ускоряет свёртки при фиксированном размере входа)
torch.backends.cudnn.benchmark = True

DEVICE = (torch.device("cuda") if torch.cuda.is_available() else
          torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cpu"))

# ← Изменено: инициализируем scaler один раз (используется для AMP)
use_amp = DEVICE.type == "cuda"
scaler = GradScaler(enabled=use_amp)


def train_step(model: nn.Module,
               dataloader: utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: optim.Optimizer,
               device=DEVICE):
    """Один проход обучения по dataloader с AMP и non_blocking copy"""
    model.train()
    train_loss, train_acc = 0.0, 0.0
    all_preds, all_targets = [], []

    for X, y in dataloader:
        X, y = X.to(device, non_blocking=use_amp), y.to(device, non_blocking=use_amp)

        # ← Изменено: используем автоматическую смешанную точность
        with (autocast(dtype=torch.float16) if use_amp else nullcontext()):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        preds = y_pred.softmax(dim=1).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
        train_acc += (preds == y).float().mean().item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc, all_preds, all_targets


def eval_step(model: nn.Module,
              dataloader: utils.data.DataLoader,
              loss_fn: nn.Module,
              device=DEVICE):
    """Оценка модели (валидация / тест) с AMP"""

    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    all_preds, all_targets = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=use_amp), y.to(device, non_blocking=use_amp)

            ctx = autocast(dtype=torch.float16) if use_amp else nullcontext()
            with ctx:
                logits = model(X)
                loss = loss_fn(logits, y)
            loss_sum += loss.item()
            preds = logits.softmax(dim=1).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
            acc_sum += (preds == y).float().mean().item()

    loss_avg = loss_sum / len(dataloader)
    acc_avg = acc_sum / len(dataloader)
    return loss_avg, acc_avg, all_preds, all_targets


def train(model: nn.Module,
          train_dataloader: utils.data.DataLoader,
          val_dataloader: utils.data.DataLoader,
          test_dataloader: utils.data.DataLoader,
          optimizer: optim.Optimizer,
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          save_path: str = 'saved_weights.pt'):
    """Тренировочный цикл с AMP, лучшим чекпоинтом и логированием в ClearML."""

    task = Task.current_task()
    logger = task.get_logger()

    # ← Изменено: сокращаем период отчёта до 5 с, чтобы Scalars почаще обновлялись
    logger.set_flush_period(5)

    results = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "test_loss":  None, "test_acc": None
    }

    best_val = float("inf")

    for epoch in tqdm(range(epochs), desc="Эпоха"):
        # 1) Обучение
        train_loss, train_acc, train_preds, train_targets = train_step(model, train_dataloader, loss_fn, optimizer)
        # 2) Валидация
        val_loss, val_acc, val_preds, val_targets = eval_step(model, val_dataloader, loss_fn)

        # 3) Логирование русскими подписями
        logger.report_scalar("Loss",   "обучение",  iteration=epoch, value=train_loss)
        logger.report_scalar("Accuracy", "обучение",  iteration=epoch, value=train_acc)
        train_f1 = f1_score(train_targets, train_preds, average="macro")
        logger.report_scalar("F1", "обучение",  iteration=epoch, value=train_f1)

        logger.report_scalar("Loss", "валидация", iteration=epoch, value=val_loss)
        logger.report_scalar("Accuracy", "валидация", iteration=epoch, value=val_acc)
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        logger.report_scalar("F1", "валидация", iteration=epoch, value=val_f1)

        # 4) Лучший чекпоинт
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            logger.report_text(f"Лучшая модель сохранена на эпохе {epoch+1} (val_loss={val_loss:.4f})")

        print(f"Эпоха {epoch+1}/{epochs} — Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f}, acc={val_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    # === Тестирование после обучения ===
    test_loss, test_acc, test_preds, test_targets = eval_step(model, test_dataloader, loss_fn)
    test_f1 = f1_score(test_targets, test_preds, average="macro")
    logger.report_scalar("Loss",   "тест", iteration=epochs, value=test_loss)
    logger.report_scalar("Accuracy", "тест", iteration=epochs, value=test_acc)
    logger.report_scalar("F1", "тест", iteration=epochs, value=test_f1)
    logger.report_text(f"Финальный тест — loss={test_loss:.4f}, acc={test_acc:.4f}")

    results["test_loss"] = test_loss
    results["test_acc"] = test_acc

    return results
