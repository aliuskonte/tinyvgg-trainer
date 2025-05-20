import sys
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import multiprocessing
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def get_data_transform(size: tuple = (64, 64)) -> transforms.Compose:
    """
    Создаёт последовательность преобразований для изображений:
    изменение размера, случайная ротация и преобразование в тензор.
    """
    return transforms.Compose([
        transforms.Resize(size=size),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.ToTensor()
    ])


def load_data(
        split_dir: str | None = None,
        transform=None):
    """
    Загружает датасеты train, val и test из указанной папки split_dir.
    :return: train_data, val_data, test_data
    """
    data_transform = transform if transform is not None else get_data_transform()

    path_split_data = Path(split_dir) if split_dir else Path("src/tinyvgg_trainer/data/split")
    path_train_ds = path_split_data / "train"
    path_val_ds = path_split_data / "val"
    path_test_ds = path_split_data / "test"

    train_data = datasets.ImageFolder(
        root=path_train_ds,
        transform=data_transform
    )
    val_data = datasets.ImageFolder(
        root=path_val_ds,
        transform=data_transform
    )
    test_data = datasets.ImageFolder(
        root=path_test_ds,
        transform=data_transform
    )

    return train_data, val_data, test_data


def create_dataloaders(
        train_data,
        val_data,
        test_data,
        batch_size: int = 32,
        num_workers: int | None = None,
        pin_memory: bool = True
):
    """
    Создаёт DataLoader'ы для train, val и test.
    :return: train_dataloader, val_dataloader, test_dataloader
    """

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count())

    logging.info(f"Number of workers: {num_workers}")

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory
    )
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    # Пример использования исходных переменных
    transform = get_data_transform((64, 64))
    train_data, val_data, test_data = load_data(transform=transform)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_data, val_data, test_data
    )
    print(f"train_dataloader batches: {len(train_dataloader)}")
    print(f"val_dataloader batches: {len(val_dataloader)}")
    print(f"test_dataloader batches: {len(test_dataloader)}")