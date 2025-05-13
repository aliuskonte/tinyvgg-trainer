import argparse
from pathlib import Path

import torch
from torch import nn
from timeit import default_timer as timer

from src.models.tiny_vgg import TinyVGG
from src.prepare_dataloaders import get_data_transform, load_data, create_dataloaders
from src.training_loop import train


def main(epochs: int, learning_rate: float, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Устанавливаем зерно для воспроизводимости
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Создаём экземпляр модели TinyVGG
    model = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(train_data.classes)
    ).to(device)

    # Функция потерь и оптимизатор
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Запускаем таймер
    start_time = timer()

    # Папка для чекпоинтов
    save_path = Path("checkpoints") / f"tinyvgg_epochs_{epochs}_checkpoint.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Обучаем модель
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        save_path=f"checkpoints/tinyvgg_epochs_{epochs}_checkpoint.pt"
    )

    # Останавливаем таймер
    end_time = timer()
    print(f"Общее время обучения: {end_time - start_time:.3f} секунд")

    # Сохраняем словарь метрик
    save_path = Path("metrics") / f"tinyvgg_epoch{epochs}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, f"metrics/tinyvgg_epoch_{epochs}_results.pt")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для обучения модели TinyVGG с этапами train/val/test")

    transform = get_data_transform((64, 64))
    train_data, val_data, test_data = load_data(transform=transform)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_data, val_data, test_data
    )
    parser.add_argument("--epochs", type=int, default=2, help="Количество эпох обучения")
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.001, help="Шаг обучения оптимизатора")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно для воспроизводимости")
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed
    )
