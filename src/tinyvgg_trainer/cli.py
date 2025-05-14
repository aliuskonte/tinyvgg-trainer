#!/usr/bin/env python3
"""
cli.py — единая точка входа для обучения TinyVGG с ClearML интеграцией

Запуск из терминала:
    python -m tinyvgg_trainer.cli --epochs 5 --lr 0.0005 --split-dir /path/to/split
или после установки пакета:
    train-tinyvgg --epochs 5 --split-dir /path/to/split
"""

import argparse
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch import nn
from clearml import Task

from tinyvgg_trainer.models.tiny_vgg import TinyVGG
from tinyvgg_trainer.prepare_dataloaders import (
    get_data_transform,
    load_data,
    create_dataloaders,
)
from tinyvgg_trainer.training_loop import train

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def run(
    epochs: int,
    lr: float,
    seed: int,
    img_size: int,
    split_dir: str | None = None,
) -> None:
    """Запускает полный цикл train / val / test с ClearML трекингом."""
    # Инициализируем задачу ClearML
    task = Task.init(
        project_name="TinyVGG-Trainer",
        task_name=f"Train_img{img_size}_lr{lr}_ep{epochs}",
        task_type=Task.TaskTypes.training,
    )
    # Логируем гиперпараметры
    task.connect({
        "epochs": epochs,
        "learning_rate": lr,
        "seed": seed,
        "img_size": img_size,
        "split_dir": split_dir or 'data/split',
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Воспроизводимость
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Подготовка данных
    transform = get_data_transform((img_size, img_size))
    train_data, val_data, test_data = load_data(
        split_dir=split_dir,
        transform=transform,
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data
    )

    # Модель, лосс и оптимизатор
    model = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(train_data.classes),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Пути для сохранения
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "tinyvgg_best.pt"

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"tinyvgg_epoch{epochs}_results.pt"

    # Обучение
    start = timer()
    results = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        save_path=str(ckpt_path),
    )
    duration = timer() - start
    logging.info(f"Обучение завершено за {duration:.2f} с")

    # Сохранение метрик
    torch.save(results, metrics_path)
    logging.info(f"Метрики сохранены в: {metrics_path}")
    logging.info(f"Лучший чекпоинт сохранён в: {ckpt_path}")

    # Логируем артефакты в ClearML
    task.upload_artifact(name="best_model", artifact_object=str(ckpt_path))
    task.upload_artifact(name="metrics", artifact_object=str(metrics_path))


def main():
    """Парсинг аргументов и запуск обучения через run()."""
    parser = argparse.ArgumentParser(
        description="Скрипт для обучения TinyVGG (train / val / test)"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Кол-во эпох")
    parser.add_argument(
        "--lr", "--learning-rate",
        dest="lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument(
        "--img-size", type=int, default=64,
        help="Размер стороны входного изображения",
    )
    parser.add_argument(
        "--split-dir", type=str, default=None,
        help="Путь к папке data/split (train/val/test)",
    )
    args = parser.parse_args()

    run(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        img_size=args.img_size,
        split_dir=args.split_dir,
    )


if __name__ == "__main__":
    main()